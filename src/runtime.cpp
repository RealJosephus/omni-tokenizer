#include "internal.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace omni {
namespace {
Value linear(Context &c,const Value&x,const std::string&name){return c.ops->linear(x,c.model->w(name+".weight"),c.model->optional(name+".bias"));}
Value image_conv(Context &c,const Value&x,const std::string&name){auto y=c.ops->linear(x,c.model->w(name+".weight"),nullptr);c.ops->bias(y,c.model->w(name+".bias"));return y;}
Value norm(Context &c,const Value&x,const std::string&name,float eps){return c.ops->norm(x,c.model->w(name+".weight"),c.model->w(name+".bias"),eps);}
void finite_input(const float*p,size_t n){for(size_t i=0;i<n;++i)require(std::isfinite(p[i]),"Input contains a non-finite value");}
void trace(Context&c,const Value&x,const std::string&name){if(c.trace){auto values=c.ops->download(x);c.trace(c.trace_user,name.c_str(),values.data(),x.rows,x.cols);}}
}
std::vector<uint16_t> audio(Context &c,const float *mel,uint32_t frames,uint32_t valid) {
    const auto &cfg=c.model->cfg; c.check();
    finite_input(mel,size_t(frames)*cfg.input_channels);
    std::vector<float> transposed(size_t(frames)*cfg.input_channels);
    for(size_t t=0;t<frames;++t)for(size_t j=0;j<cfg.input_channels;++j)transposed[t*cfg.input_channels+j]=mel[j*frames+t];
    auto x=c.ops->upload(transposed.data(),frames,cfg.input_channels,false);
    x=linear(c,c.ops->conv_gather(x,1),"conv1");c.ops->gelu(x);
    x=linear(c,c.ops->conv_gather(x,2),"conv2");c.ops->gelu(x);
    c.ops->add(x,c.model->w("embed_positions.weight"));
    trace(c,x,"embedding");
    for(uint32_t i=0;i<cfg.layers;++i){c.check();auto p="layers."+std::to_string(i)+".";
        auto z=norm(c,x,p+"self_attn_layer_norm",1e-5f);
        auto q=linear(c,z,p+"self_attn.q_proj"),k=linear(c,z,p+"self_attn.k_proj"),v=linear(c,z,p+"self_attn.v_proj");
        z=c.ops->attention(q,&k,&v,cfg.heads,cfg.block_size,(valid+1)/2,false);
        z=linear(c,z,p+"self_attn.out_proj");c.ops->add(x,z);
        z=norm(c,x,p+"final_layer_norm",1e-5f);z=linear(c,z,p+"fc1");c.ops->gelu(z);z=linear(c,z,p+"fc2");c.ops->add(x,z);
        trace(c,x,"layer."+std::to_string(i));
    }
    c.check();c.features=c.ops->pool(x,cfg.pool_size);c.features.rows=(valid+7)/8;
    auto out=c.ops->quantize(c.features,c.model->w("codebook.weight"),c.model->codebook_norms,c.cancelled);c.ops->synchronize();c.check();return out;
}
std::vector<uint16_t> image(Context &c,const float*pixels,uint32_t height,uint32_t width) {
    const auto &cfg=c.model->cfg;c.check();finite_input(pixels,size_t(height)*width*3);
    size_t p=cfg.patch_size,gh=height/p,gw=width/p,n=gh*gw,pd=8*p*p;
    std::vector<float> patches(n*pd);
    for(size_t y=0;y<gh;++y)for(size_t x=0;x<gw;++x)for(size_t ch=0;ch<3;++ch)for(size_t dy=0;dy<p;++dy)for(size_t dx=0;dx<p;++dx)patches[(y*gw+x)*pd+(dy*p+dx)*8+ch]=pixels[ch*height*width+(y*p+dy)*width+x*p+dx];
    auto x=c.ops->upload(patches.data(),n,pd,true);x=image_conv(c,x,"patch_embed.proj");trace(c,x,"patch");
    auto pos=c.ops->interpolate(c.model->w("pos_embed"),cfg.pos_height,gh,gw);trace(c,pos,"position");c.ops->add(x,pos);pos={};trace(c,x,"embedding");
    for(uint32_t i=0;i<cfg.layers;++i){c.check();auto pre="blocks."+std::to_string(i)+".";
        auto z=norm(c,x,pre+"norm1",1e-6f);if(i==0)trace(c,z,"norm1.0");z=linear(c,z,pre+"attn.qkv");if(i==0)trace(c,z,"qkv.0");
        z=c.ops->attention(z,nullptr,nullptr,cfg.heads,0,n,true);if(i==0)trace(c,z,"attention.0");z=linear(c,z,pre+"attn.proj");c.ops->add(x,z);if(i==0)trace(c,x,"residual1.0");
        z=norm(c,x,pre+"norm2",1e-6f);z=linear(c,z,pre+"mlp.fc1");if(i==0)trace(c,z,"fc1.0");c.ops->gelu(z);if(i==0)trace(c,z,"gelu.0");z=linear(c,z,pre+"mlp.fc2");if(i==0)trace(c,z,"fc2.0");c.ops->add(x,z);
        trace(c,x,"layer."+std::to_string(i));
    }
    c.check();x=image_conv(c,x,"quant_conv");trace(c,x,"projected");c.features=c.ops->normalize(x);
    auto out=c.ops->quantize(c.features,c.model->w("quantize.embedding.weight"),c.model->codebook_norms,c.cancelled);c.ops->synchronize();c.check();return out;
}
}
struct omni_model {std::shared_ptr<omni::Model> impl;};
struct omni_context {omni::Context impl;};
namespace {
thread_local std::string error;
template<class F> int protect(F f){try{error.clear();f();return OMNI_OK;}catch(const omni::Error&e){error=e.what();return e.code;}catch(const std::bad_alloc&){error="Allocation failed";return OMNI_OUT_OF_MEMORY;}catch(const std::exception&e){error=e.what();return OMNI_INTERNAL_ERROR;}catch(...){error="Unknown native failure";return OMNI_INTERNAL_ERROR;}}
struct Running {omni::Context&c; explicit Running(omni::Context&ctx):c(ctx){omni::require(!c.running.test_and_set(),"Context is already executing");}~Running(){c.running.clear();}};
}
extern "C" {
const char *omni_last_error(void){return error.c_str();}
const char *omni_version(void){return "0.1.0";}
int omni_has_cuda(void){
#ifdef OMNI_CUDA_ENABLED
return 1;
#else
return 0;
#endif
}
int omni_model_load(const char*path,int backend,int device,omni_model**out){if(out)*out=nullptr;return protect([&]{omni::require(out,"Output handle is null");auto m=omni::load(path,backend,device);*out=new omni_model{std::move(m)};});}
void omni_model_free(omni_model*m){delete m;}
int omni_model_get_info(const omni_model*m,omni_model_info*out){return protect([&]{omni::require(m&&out&&out->struct_size>=sizeof(omni_model_info),"Invalid model info arguments");auto &c=m->impl->cfg;*out={sizeof(omni_model_info),c.kind,c.layers,c.width,c.heads,c.code_dim,c.vocab_size,uint32_t(m->impl->backend),m->impl->weight_bytes};});}
int omni_context_create(omni_model*m,omni_context**out){if(out)*out=nullptr;return protect([&]{omni::require(m&&out,"Invalid context arguments");auto c=std::make_unique<omni_context>();c->impl.model=m->impl;c->impl.ops=omni::make_ops(m->impl->backend,m->impl->device);c->impl.ops->enable_reuse();*out=c.release();});}
void omni_context_free(omni_context*c){delete c;}
void omni_context_cancel(omni_context*c){if(c)c->impl.cancelled.store(true);}
void omni_context_reset_cancel(omni_context*c){if(c)c->impl.cancelled.store(false);}
int omni_context_set_trace(omni_context*c,omni_trace_callback callback,void*user){return protect([&]{omni::require(c,"Invalid context");Running guard(c->impl);c->impl.trace=callback;c->impl.trace_user=user;});}
int omni_encode_audio(omni_context*c,const float*mel,uint32_t frames,uint32_t valid,uint16_t*codes,size_t capacity,size_t*count){return protect([&]{omni::require(c&&mel&&count,"Invalid audio pointers");omni::require(c->impl.model->cfg.kind==OMNI_AUDIO,"Model is not an audio tokenizer");omni::require(frames>0&&frames%8==0&&frames<=3000&&frames/2<=c->impl.model->cfg.pos_height&&valid>0&&valid<=frames,"Invalid audio frame lengths");*count=(valid+7)/8;omni::require(codes&&capacity>=*count,"Output capacity is too small");Running guard(c->impl);c->impl.features={};auto result=omni::audio(c->impl,mel,frames,valid);std::copy(result.begin(),result.end(),codes);});}
int omni_encode_image(omni_context*c,const float*pixels,uint32_t height,uint32_t width,uint16_t*codes,size_t capacity,size_t*count){return protect([&]{omni::require(c&&pixels&&count,"Invalid image pointers");auto &cfg=c->impl.model->cfg;omni::require(cfg.kind==OMNI_IMAGE,"Model is not an image tokenizer");omni::require(height>0&&width>0&&height<=1152&&width<=1152&&height%cfg.patch_size==0&&width%cfg.patch_size==0,"Invalid image dimensions");*count=size_t(height/cfg.patch_size)*(width/cfg.patch_size);omni::require(codes&&capacity>=*count,"Output capacity is too small");Running guard(c->impl);c->impl.features={};auto result=omni::image(c->impl,pixels,height,width);std::copy(result.begin(),result.end(),codes);});}
int omni_context_copy_features(omni_context*c,float*out,size_t capacity,size_t*count){return protect([&]{omni::require(c&&count,"Invalid feature pointers");Running guard(c->impl);*count=c->impl.features.size();omni::require(*count>0,"No completed feature tensor");omni::require(out&&capacity>=*count,"Feature output capacity is too small");auto data=c->impl.ops->download(c->impl.features);std::copy(data.begin(),data.end(),out);});}
uint64_t omni_context_peak_workspace(const omni_context*c){return c?c->impl.ops->memory->peak.load():0;}
}
