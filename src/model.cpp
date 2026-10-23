#include "internal.h"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>

namespace omni {
namespace {
uint32_t u32(std::istream &f) { unsigned char b[4]; f.read(reinterpret_cast<char*>(b),4); require(bool(f),"Truncated model index",OMNI_INVALID_MODEL); return uint32_t(b[0])|(uint32_t(b[1])<<8)|(uint32_t(b[2])<<16)|(uint32_t(b[3])<<24); }
uint64_t u64(std::istream &f) { auto lo=u32(f); auto hi=u32(f); return lo|(uint64_t(hi)<<32); }
struct Record {std::string name; uint32_t dtype; uint64_t offset,bytes; std::vector<uint64_t> shape;};
void shape(const Model &m,const std::string &name,std::initializer_list<uint64_t> s) {
    auto i=m.tensors.find(name); require(i!=m.tensors.end(),"Missing tensor: "+name,OMNI_INVALID_MODEL);
    require(i->second.shape==std::vector<uint64_t>(s),"Incorrect shape for "+name,OMNI_INVALID_MODEL);
}
void validate(const Model &m) {
    const auto &c=m.cfg;
    if(c.kind==OMNI_AUDIO) {
        shape(m,"conv1.weight",{c.width,c.input_channels,3}); shape(m,"conv1.bias",{c.width});
        shape(m,"conv2.weight",{c.width,c.width,3}); shape(m,"conv2.bias",{c.width});
        shape(m,"embed_positions.weight",{c.pos_height,c.width}); shape(m,"codebook.weight",{c.vocab_size,c.width});
        require(c.pool_size==4&&c.patch_size==0&&c.code_dim==c.width&&c.block_size>0,"Invalid audio configuration",OMNI_INVALID_MODEL);
        for(uint32_t l=0;l<c.layers;++l) {
            auto p="layers."+std::to_string(l)+".";
            for(auto n:{"self_attn_layer_norm","final_layer_norm"}) {shape(m,p+n+".weight",{c.width});shape(m,p+n+".bias",{c.width});}
            for(auto n:{"q_proj","k_proj","v_proj","out_proj"}) shape(m,p+"self_attn."+n+".weight",{c.width,c.width});
            for(auto n:{"q_proj","v_proj","out_proj"}) shape(m,p+"self_attn."+n+".bias",{c.width});
            shape(m,p+"fc1.weight",{c.ffn,c.width}); shape(m,p+"fc1.bias",{c.ffn}); shape(m,p+"fc2.weight",{c.width,c.ffn}); shape(m,p+"fc2.bias",{c.width});
        }
    } else {
        require(c.input_channels==3&&c.patch_size>0&&c.patch_size<=64&&c.pos_height==c.pos_width,"Invalid image configuration",OMNI_INVALID_MODEL);
        shape(m,"patch_embed.proj.weight",{c.width,3,c.patch_size,c.patch_size});shape(m,"patch_embed.proj.bias",{c.width});
        shape(m,"pos_embed",{1,uint64_t(c.pos_height)*c.pos_width,c.width});
        shape(m,"quant_conv.weight",{c.code_dim,c.width,1,1}); shape(m,"quant_conv.bias",{c.code_dim});shape(m,"quantize.embedding.weight",{c.vocab_size,c.code_dim});
        for(uint32_t l=0;l<c.layers;++l) {
            auto p="blocks."+std::to_string(l)+".";
            for(auto n:{"norm1","norm2"}) {shape(m,p+n+".weight",{c.width});shape(m,p+n+".bias",{c.width});}
            shape(m,p+"attn.qkv.weight",{3*c.width,c.width});shape(m,p+"attn.qkv.bias",{3*c.width});
            shape(m,p+"attn.proj.weight",{c.width,c.width});shape(m,p+"attn.proj.bias",{c.width});
            shape(m,p+"mlp.fc1.weight",{c.ffn,c.width});shape(m,p+"mlp.fc1.bias",{c.ffn});shape(m,p+"mlp.fc2.weight",{c.width,c.ffn});shape(m,p+"mlp.fc2.bias",{c.width});
        }
    }
}
}
std::unique_ptr<Ops> make_ops(int backend,int device) {
    if(backend==OMNI_CPU) return make_cpu();
#ifdef OMNI_CUDA_ENABLED
    if(backend==OMNI_CUDA) return make_cuda(device);
#else
    (void)device;
#endif
    throw Error(OMNI_UNSUPPORTED,"Requested backend is unavailable in this build");
}
std::shared_ptr<Model> load(const char *path,int backend,int device) {
    require(path&&*path,"Model path must not be empty");
    std::ifstream f(std::filesystem::u8path(path),std::ios::binary|std::ios::ate);
    require(bool(f),"Cannot open model file",OMNI_IO_ERROR);
    auto end=f.tellg(); require(end>=68,"Model file is too short",OMNI_INVALID_MODEL);
    uint64_t file_bytes=static_cast<uint64_t>(end); f.seekg(0);
    char magic[8]; f.read(magic,8); require(std::memcmp(magic,"OMNIVQ01",8)==0,"Invalid model magic",OMNI_INVALID_MODEL);
    require(u32(f)==1,"Unsupported model version",OMNI_INVALID_MODEL);
    auto m=std::make_shared<Model>(); m->backend=backend;m->device=device;
    auto &c=m->cfg;c.kind=u32(f);
    uint32_t *fields[]={&c.layers,&c.width,&c.heads,&c.ffn,&c.input_channels,&c.patch_size,&c.pool_size,&c.block_size,&c.pos_height,&c.pos_width,&c.code_dim,&c.vocab_size};
    for(auto p:fields)*p=u32(f);
    require((c.kind==1||c.kind==2)&&c.layers>0&&c.layers<=80&&c.width>0&&c.width<=8192&&c.heads>0&&c.width%c.heads==0&&c.ffn>0&&c.ffn<=32768&&c.code_dim>0&&c.code_dim<=8192&&c.vocab_size>0&&c.vocab_size<=16384&&c.input_channels>0&&c.input_channels<=512&&c.pos_height>0&&c.pos_height<=16384&&c.pos_width<=16384,"Model configuration exceeds supported bounds",OMNI_INVALID_MODEL);
    uint32_t count=u32(f);require(count==6+c.layers*(c.kind==OMNI_AUDIO?15:12),"Unexpected tensor count for the fixed model",OMNI_INVALID_MODEL);
    std::vector<Record> records;std::set<std::string> names;
    for(uint32_t i=0;i<count;++i) {
        uint32_t len=u32(f),dtype=u32(f),rank=u32(f),reserved=u32(f);Record r; r.dtype=dtype;r.offset=u64(f);r.bytes=u64(f);
        require(len>0&&len<=1024&&dtype==(c.kind==OMNI_AUDIO?1U:2U)&&rank>0&&rank<=4&&reserved==0,"Invalid tensor record or precision profile",OMNI_INVALID_MODEL);
        uint64_t elements=1;for(uint32_t j=0;j<rank;++j) {auto d=u64(f);require(d>0&&d<=uint64_t(1)<<24&&elements<=std::numeric_limits<uint64_t>::max()/d,"Tensor dimensions overflow",OMNI_INVALID_MODEL);elements*=d;r.shape.push_back(d);}
        require(elements<=std::numeric_limits<uint64_t>::max()/4&&r.bytes==elements*(dtype==1?4:2)&&r.offset%64==0&&r.offset<=file_bytes&&r.bytes<=file_bytes-r.offset,"Invalid tensor data bounds",OMNI_INVALID_MODEL);
        r.name.resize(len);f.read(&r.name[0],len);require(bool(f)&&r.name.find('\0')==std::string::npos&&names.insert(r.name).second,"Invalid or duplicate tensor name",OMNI_INVALID_MODEL); records.push_back(std::move(r));
    }
    auto index_end=static_cast<uint64_t>(f.tellg());auto sorted=records;
    std::sort(sorted.begin(),sorted.end(),[](const Record&a,const Record&b){return a.offset<b.offset;});uint64_t previous=index_end;
    for(auto &r:sorted){require(r.offset>=previous,"Overlapping model tensors/index",OMNI_INVALID_MODEL);previous=r.offset+r.bytes;}
    for(const auto&r:records)m->tensors.emplace(r.name,Tensor{{},r.shape});
    validate(*m);
    auto ops=make_ops(backend,device);
    for(auto &r:records) {
        std::vector<unsigned char> raw(static_cast<size_t>(r.bytes)); f.seekg(static_cast<std::streamoff>(r.offset));f.read(reinterpret_cast<char*>(raw.data()),static_cast<std::streamsize>(r.bytes));require(bool(f),"Truncated tensor data",OMNI_IO_ERROR);
        size_t rows=static_cast<size_t>(r.shape[0]),cols=1;for(size_t j=1;j<r.shape.size();++j)cols*=static_cast<size_t>(r.shape[j]);
        if(r.name=="pos_embed"){rows=static_cast<size_t>(r.shape[1]);cols=static_cast<size_t>(r.shape[2]);}
        if(c.kind==OMNI_IMAGE&&r.name=="patch_embed.proj.weight"){
            const size_t pixels=size_t(c.patch_size)*c.patch_size;
            std::vector<unsigned char> reordered(rows*pixels*8*2,0);
            for(size_t o=0;o<rows;++o)for(size_t px=0;px<pixels;++px)for(size_t ch=0;ch<3;++ch){size_t from=(o*3*pixels+ch*pixels+px)*2,to=(o*pixels*8+px*8+ch)*2;reordered[to]=raw[from];reordered[to+1]=raw[from+1];}
            raw.swap(reordered);cols=pixels*8;
        }
        auto value=ops->raw_upload(raw.data(),rows,cols,r.dtype==2);m->weight_bytes+=value.size()*(backend==OMNI_CPU?4:(value.bf16?2:4));m->tensors.at(r.name).value=std::move(value);
    }
    const char *cb=c.kind==OMNI_AUDIO?"codebook.weight":"quantize.embedding.weight";
    if(c.kind==OMNI_IMAGE)m->tensors.at(cb).value=ops->normalize(m->w(cb));
    m->codebook_norms=ops->squares(m->w(cb));ops->synchronize();
    return m;
}
}
