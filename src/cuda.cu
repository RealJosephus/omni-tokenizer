#include "internal.h"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <algorithm>
#include <cmath>
#include <map>
#include <mutex>
#include <tuple>

namespace omni {
namespace {
void check_cuda(cudaError_t e){if(e!=cudaSuccess)throw Error(e==cudaErrorMemoryAllocation?OMNI_OUT_OF_MEMORY:OMNI_INTERNAL_ERROR,std::string("CUDA: ")+cudaGetErrorString(e));}
void check_blas(cublasStatus_t e){if(e!=CUBLAS_STATUS_SUCCESS)throw Error(OMNI_INTERNAL_ERROR,"cuBLAS status "+std::to_string(int(e)));}
__device__ float get(const void*p,size_t i,bool b){return b?__bfloat162float(static_cast<const __nv_bfloat16*>(p)[i]):static_cast<const float*>(p)[i];}
__device__ void put(void*p,size_t i,float v,bool b){if(b)static_cast<__nv_bfloat16*>(p)[i]=__float2bfloat16_rn(v);else static_cast<float*>(p)[i]=v;}
__device__ float rnd(float x,bool b){return b?__bfloat162float(__float2bfloat16_rn(x)):x;}
__device__ float sum_reduce(float x){__shared__ float s[256];s[threadIdx.x]=x;__syncthreads();for(int stride=128;stride;stride>>=1){if(threadIdx.x<stride)s[threadIdx.x]+=s[threadIdx.x+stride];__syncthreads();}float result=s[0];__syncthreads();return result;}
__device__ float max_reduce(float x){__shared__ float s[256];s[threadIdx.x]=x;__syncthreads();for(int stride=128;stride;stride>>=1){if(threadIdx.x<stride)s[threadIdx.x]=fmaxf(s[threadIdx.x],s[threadIdx.x+stride]);__syncthreads();}float result=s[0];__syncthreads();return result;}
__global__ void convert_kernel(const float*x,void*y,size_t n,bool b){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)put(y,i,x[i],b);}
__global__ void download_kernel(const void*x,float*y,size_t n,bool b){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)y[i]=get(x,i,b);}
__global__ void gelu_kernel(void*x,size_t n,bool b){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n){float z=get(x,i,b);put(x,i,.5f*z*(1+erff(z*.7071067811865475244f)),b);}}
__global__ void add_kernel(void*x,const void*y,size_t n,bool xb,bool yb){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)put(x,i,get(x,i,xb)+get(y,i,yb),xb);}
__global__ void bias_kernel(void*x,const void*b,size_t n,size_t d,bool bf,bool bb){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)put(x,i,get(x,i,bf)+get(b,i%d,bb),bf);}
__global__ void norm_kernel(const void*x,void*y,const void*w,const void*b,size_t d,float eps,bool bf,bool wb,bool bb){size_t row=blockIdx.x;float sum=0;for(size_t j=threadIdx.x;j<d;j+=256)sum+=get(x,row*d+j,bf);float mean=sum_reduce(sum)/float(d);sum=0;for(size_t j=threadIdx.x;j<d;j+=256){float z=get(x,row*d+j,bf)-mean;sum+=z*z;}float inv=rsqrtf(sum_reduce(sum)/float(d)+eps);for(size_t j=threadIdx.x;j<d;j+=256)put(y,row*d+j,(get(x,row*d+j,bf)-mean)*inv*get(w,j,wb)+get(b,j,bb),bf);}
struct Moments {float mean,variance,count;};
__device__ Moments merge(Moments a,Moments b){float count=a.count+b.count;if(count==0)return {0,0,0};float delta=b.mean-a.mean,inv=1/count,wa=a.count*inv,wb=b.count*inv;return {wa*a.mean+wb*b.mean,a.variance+b.variance+delta*delta*a.count*wb,count};}
__global__ void norm_welford_kernel(const void*x,void*y,const void*w,const void*b,size_t d,float eps){
    size_t row=blockIdx.x;unsigned tid=threadIdx.x;Moments m{0,0,0};
    for(size_t base=tid*4;base<d;base+=512)for(size_t j=base;j<base+4;++j){float z=get(x,row*d+j,true),delta=z-m.mean,count=m.count+1,mean=m.mean+delta*(1/count);m={mean,m.variance+delta*(z-mean),count};}
    for(int step=16;step;step/=2){Moments other{__shfl_down_sync(0xffffffff,m.mean,step),__shfl_down_sync(0xffffffff,m.variance,step),__shfl_down_sync(0xffffffff,m.count,step)};m=merge(other,m);}
    __shared__ Moments warps[4];if(tid%32==0)warps[tid/32]=m;__syncthreads();
    for(int step=2;step;step/=2){if(tid<unsigned(step))warps[tid]=merge(warps[tid+step],warps[tid]);__syncthreads();}
    float mean=warps[0].mean,inv=rsqrtf(warps[0].variance/float(d)+eps);
    for(size_t j=tid;j<d;j+=128)put(y,row*d+j,get(w,j,true)*(inv*(get(x,row*d+j,true)-mean))+get(b,j,true),true);
}
__global__ void gather_kernel(const void*x,void*y,size_t rows,size_t d,size_t stride,bool bf){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=rows*d*3)return;size_t t=i/(d*3),c=(i/3)%d,k=i%3;int64_t src=int64_t(t*stride+k)-2;put(y,i,src<0?0:get(x,size_t(src)*d+c,bf),bf);}
__global__ void pool_kernel(const void*x,void*y,size_t n,size_t rows,size_t d,size_t factor,bool bf){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;size_t row=i/d,col=i%d;float sum=0;for(size_t k=0;k<factor&&row*factor+k<rows;++k)sum+=get(x,(row*factor+k)*d+col,bf);put(y,i,sum/float(factor),bf);}
__global__ void position_kernel(const void*x,void*y,size_t sh,size_t h,size_t w,size_t d,bool bf){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=h*w*d)return;size_t row=i/d/w,col=(i/d)%w,c=i%d;float sy=fmaxf(0,(float(row)+.5f)*(float(sh)/float(h))-.5f),sx=fmaxf(0,(float(col)+.5f)*(float(sh)/float(w))-.5f);size_t y0=size_t(sy),x0=size_t(sx),y1=min(y0+1,sh-1),x1=min(x0+1,sh-1);float fy=sy-float(y0),fx=sx-float(x0);float a=__fmaf_rn(1-fx,get(x,(y0*sh+x0)*d+c,bf),__fmul_rn(fx,get(x,(y0*sh+x1)*d+c,bf)));float b=__fmaf_rn(1-fx,get(x,(y1*sh+x0)*d+c,bf),__fmul_rn(fx,get(x,(y1*sh+x1)*d+c,bf)));put(y,i,__fmaf_rn(1-fy,a,__fmul_rn(fy,b)),bf);}
__global__ void normalize_kernel(const void*x,void*y,size_t d,bool bf){size_t row=blockIdx.x;float sum=0;for(size_t j=threadIdx.x;j<d;j+=256){float z=get(x,row*d+j,bf);sum+=z*z;}float norm=fmaxf(rnd(sqrtf(sum_reduce(sum)),bf),rnd(1e-12f,bf));for(size_t j=threadIdx.x;j<d;j+=256)put(y,row*d+j,get(x,row*d+j,bf)/norm,bf);}
__global__ void squares_kernel(const void*x,void*y,size_t d,bool bf){size_t row=blockIdx.x;float sum=0;for(size_t j=threadIdx.x;j<d;j+=256){float z=get(x,row*d+j,bf);sum+=rnd(z*z,bf);}float result=sum_reduce(sum);if(threadIdx.x==0)put(y,row,result,bf);}
__global__ void pack_kernel(const void*q,const void*k,const void*v,float*qp,float*kp,float*vp,size_t n,size_t d,size_t heads,bool bf,bool fused,float factor){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n*d)return;size_t hd=d/heads,h=i/(n*hd),t=(i/hd)%n,j=i%hd;size_t source=t*(fused?3*d:d)+h*hd+j;float qv=get(q,source,bf);qp[i]=qv*factor;kp[i]=fused?get(q,source+d,bf)*factor:get(k,t*d+h*hd+j,bf);vp[i]=fused?get(q,source+2*d,bf):get(v,t*d+h*hd+j,bf);}
__global__ void softmax_kernel(float*s,size_t n,size_t tile,size_t base,size_t block,size_t valid){size_t row=blockIdx.x,t=row%tile+base,limit=min(n,valid);if(block)limit=min(limit,(t/block+1)*block);float mx=-CUDART_INF_F;for(size_t j=threadIdx.x;j<limit;j+=256)mx=fmaxf(mx,s[row*n+j]);mx=max_reduce(mx);float total=0;for(size_t j=threadIdx.x;j<n;j+=256){float z=j<limit?expf(s[row*n+j]-mx):0;s[row*n+j]=z;total+=z;}total=sum_reduce(total);for(size_t j=threadIdx.x;j<n;j+=256)s[row*n+j]/=total;}
__global__ void unpack_kernel(const float*x,void*y,size_t n,size_t d,size_t heads,size_t tile,size_t base,bool bf){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=tile*d)return;size_t hd=d/heads,h=i/(tile*hd),t=(i/hd)%tile,j=i%hd;put(y,(base+t)*d+h*hd+j,x[i],bf);}
__global__ void best_init(float*best,uint16_t*ids,size_t n){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n){best[i]=CUDART_INF_F;ids[i]=0;}}
__global__ void argmin_kernel(const void*dots,const void*xn,const void*cn,float*best,uint16_t*ids,size_t m,size_t tile,size_t base,bool bf){size_t row=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(row>=m)return;float score=best[row];uint16_t id=ids[row];for(size_t j=0;j<tile;++j){float dot=get(dots,row*tile+j,bf),x=get(xn,row,bf),c=get(cn,base+j,bf);float dist=bf?rnd(rnd(x+c,true)-rnd(2*dot,true),true):fmaf(-2.f,dot,x+c);if(dist<score){score=dist;id=uint16_t(base+j);}}best[row]=score;ids[row]=id;}

struct Pool {
    int device;
    std::shared_ptr<Memory> memory;
    std::multimap<size_t,void*> free;
    size_t cached=0;
    std::mutex mutex;
    static constexpr size_t limit=128*1024*1024;
    Pool(int d,std::shared_ptr<Memory> m):device(d),memory(std::move(m)){}
    ~Pool(){cudaSetDevice(device);for(auto &p:free){cudaFree(p.second);memory->live.fetch_sub(p.first);}}
    std::pair<void*,size_t> acquire(size_t bytes){
        std::lock_guard<std::mutex> guard(mutex);
        auto it=free.lower_bound(bytes);
        if(it!=free.end()&&it->first<=bytes+bytes/4){auto result=std::make_pair(it->second,it->first);cached-=it->first;free.erase(it);return result;}
        void *p=nullptr;check_cuda(cudaMalloc(&p,bytes));memory->add(bytes);return {p,bytes};
    }
    void release(void *p,size_t bytes) noexcept {std::lock_guard<std::mutex> guard(mutex);if(cached+bytes<=limit){try{free.emplace(bytes,p);cached+=bytes;return;}catch(...){}}cudaSetDevice(device);cudaFree(p);memory->live.fetch_sub(bytes);}
};

class Cuda final : public Ops {
    int device_;cudaStream_t stream_=nullptr;cublasHandle_t blas_=nullptr;cublasLtHandle_t lt_=nullptr;void*workspace_=nullptr;
    static constexpr size_t workspace_bytes=32*1024*1024;
    using Key=std::tuple<size_t,size_t,size_t,bool,bool>;
    std::map<Key,cublasLtMatmulAlgo_t> algorithms_;
    std::shared_ptr<Pool> pool_;
    Value allocate(size_t rows,size_t cols,bool bf){check_cuda(cudaSetDevice(device_));size_t bytes=rows*cols*(bf?2:4);if(pool_){auto pool=pool_;auto allocation=pool->acquire(bytes);return {std::shared_ptr<void>(allocation.first,[pool,n=allocation.second](void*q){pool->release(q,n);}),rows,cols,bf};}void*p=nullptr;check_cuda(cudaMalloc(&p,bytes));auto mem=memory;mem->add(bytes);int dev=device_;return {std::shared_ptr<void>(p,[mem,bytes,dev](void*q){cudaSetDevice(dev);cudaFree(q);mem->live.fetch_sub(bytes);}),rows,cols,bf};}
    static unsigned blocks(size_t n){return static_cast<unsigned>((n+255)/256);}
    void launch_check(){check_cuda(cudaGetLastError());}
public:
    explicit Cuda(int device):device_(device){check_cuda(cudaSetDevice(device));check_cuda(cudaStreamCreateWithFlags(&stream_,cudaStreamNonBlocking));check_blas(cublasCreate(&blas_));check_blas(cublasSetStream(blas_,stream_));check_blas(cublasSetMathMode(blas_,CUBLAS_PEDANTIC_MATH));check_blas(cublasLtCreate(&lt_));check_cuda(cudaMalloc(&workspace_,workspace_bytes));memory->add(workspace_bytes);}
    void enable_reuse() override {pool_=std::make_shared<Pool>(device_,memory);}
    ~Cuda() override {cudaSetDevice(device_);if(stream_)cudaStreamSynchronize(stream_);if(workspace_)cudaFree(workspace_);if(lt_)cublasLtDestroy(lt_);if(blas_)cublasDestroy(blas_);if(stream_)cudaStreamDestroy(stream_);}
    Value raw_upload(const void*x,size_t rows,size_t cols,bool b) override {auto y=allocate(rows,cols,b);check_cuda(cudaMemcpyAsync(y.data(),x,y.size()*(b?2:4),cudaMemcpyHostToDevice,stream_));synchronize();return y;}
    Value upload(const float*x,size_t rows,size_t cols,bool b) override {if(!b)return raw_upload(x,rows,cols,false);auto fp=raw_upload(x,rows,cols,false);auto y=allocate(rows,cols,b);convert_kernel<<<blocks(y.size()),256,0,stream_>>>(static_cast<float*>(fp.data()),y.data(),y.size(),b);launch_check();return y;}
    std::vector<float> download(const Value&x) override {std::vector<float> y(x.size());if(!x.bf16){check_cuda(cudaMemcpyAsync(y.data(),x.data(),y.size()*4,cudaMemcpyDeviceToHost,stream_));}else{auto fp=allocate(x.rows,x.cols,false);download_kernel<<<blocks(x.size()),256,0,stream_>>>(x.data(),static_cast<float*>(fp.data()),x.size(),true);launch_check();check_cuda(cudaMemcpyAsync(y.data(),fp.data(),y.size()*4,cudaMemcpyDeviceToHost,stream_));synchronize();}synchronize();return y;}
    Value linear(const Value&x,const Value&w,const Value*b) override {
        require(x.cols==w.cols&&x.bf16==w.bf16,"CUDA linear shape/dtype mismatch",OMNI_INVALID_MODEL);auto y=allocate(x.rows,w.rows,x.bf16);
        cublasLtMatmulDesc_t op=nullptr;cublasLtMatrixLayout_t a=nullptr,bb=nullptr,cc=nullptr;cublasLtMatmulPreference_t pref=nullptr;
        struct Cleanup{cublasLtMatmulDesc_t*op;cublasLtMatrixLayout_t*a,*b,*c;cublasLtMatmulPreference_t*p;~Cleanup(){if(*p)cublasLtMatmulPreferenceDestroy(*p);if(*c)cublasLtMatrixLayoutDestroy(*c);if(*b)cublasLtMatrixLayoutDestroy(*b);if(*a)cublasLtMatrixLayoutDestroy(*a);if(*op)cublasLtMatmulDescDestroy(*op);}} cleanup{&op,&a,&bb,&cc,&pref};
        auto type=x.bf16?CUDA_R_16BF:CUDA_R_32F;auto compute=x.bf16?CUBLAS_COMPUTE_32F:CUBLAS_COMPUTE_32F_PEDANTIC;
        check_blas(cublasLtMatmulDescCreate(&op,compute,CUDA_R_32F));cublasOperation_t trans=CUBLAS_OP_T;check_blas(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&trans,sizeof(trans)));
        if(b){require(b->bf16==x.bf16,"CUDA bias dtype mismatch",OMNI_INVALID_MODEL);auto epi=CUBLASLT_EPILOGUE_BIAS;const void*ptr=b->data();check_blas(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_EPILOGUE,&epi,sizeof(epi)));check_blas(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_BIAS_POINTER,&ptr,sizeof(ptr)));}
        check_blas(cublasLtMatrixLayoutCreate(&a,type,w.cols,w.rows,int64_t(w.cols)));check_blas(cublasLtMatrixLayoutCreate(&bb,type,x.cols,x.rows,int64_t(x.cols)));check_blas(cublasLtMatrixLayoutCreate(&cc,type,w.rows,x.rows,int64_t(w.rows)));
        Key key{x.rows,w.rows,x.cols,x.bf16,b!=nullptr};auto found=algorithms_.find(key);cublasLtMatmulAlgo_t algo;
        if(found==algorithms_.end()){check_blas(cublasLtMatmulPreferenceCreate(&pref));size_t ws=workspace_bytes;check_blas(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&ws,sizeof(ws)));uint32_t reduction=CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;check_blas(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,&reduction,sizeof(reduction)));cublasLtMatmulHeuristicResult_t heuristic{};int returned=0;check_blas(cublasLtMatmulAlgoGetHeuristic(lt_,op,a,bb,cc,cc,pref,1,&heuristic,&returned));require(returned>0,"cuBLASLt has no compatible algorithm",OMNI_UNSUPPORTED);algo=heuristic.algo;algorithms_.emplace(key,algo);}else algo=found->second;
        float alpha=1,beta=0;check_blas(cublasLtMatmul(lt_,op,&alpha,w.data(),a,x.data(),bb,&beta,y.data(),cc,y.data(),cc,&algo,workspace_,workspace_bytes,stream_));return y;
    }
    Value norm(const Value&x,const Value&w,const Value&b,float eps) override {auto y=allocate(x.rows,x.cols,x.bf16);if(x.bf16&&x.cols%4==0)norm_welford_kernel<<<unsigned(x.rows),128,0,stream_>>>(x.data(),y.data(),w.data(),b.data(),x.cols,eps);else norm_kernel<<<unsigned(x.rows),256,0,stream_>>>(x.data(),y.data(),w.data(),b.data(),x.cols,eps,x.bf16,w.bf16,b.bf16);launch_check();return y;}
    void gelu(Value&x) override {gelu_kernel<<<blocks(x.size()),256,0,stream_>>>(x.data(),x.size(),x.bf16);launch_check();}
    void add(Value&x,const Value&y) override {require(x.cols==y.cols&&y.rows>=x.rows,"CUDA residual dimensions disagree",OMNI_INTERNAL_ERROR);add_kernel<<<blocks(x.size()),256,0,stream_>>>(x.data(),y.data(),x.size(),x.bf16,y.bf16);launch_check();}
    void bias(Value&x,const Value&b) override {require(b.size()==x.cols,"CUDA bias dimensions disagree",OMNI_INTERNAL_ERROR);bias_kernel<<<blocks(x.size()),256,0,stream_>>>(x.data(),b.data(),x.size(),x.cols,x.bf16,b.bf16);launch_check();}
    Value conv_gather(const Value&x,size_t stride) override {auto y=allocate((x.rows+stride-1)/stride,x.cols*3,x.bf16);gather_kernel<<<blocks(y.size()),256,0,stream_>>>(x.data(),y.data(),y.rows,x.cols,stride,x.bf16);launch_check();return y;}
    Value attention(const Value&q,const Value*k,const Value*v,size_t heads,size_t block,size_t valid,bool fused) override {
        const size_t n=q.rows,d=fused?q.cols/3:q.cols,hd=d/heads;auto y=allocate(n,d,q.bf16);auto qp=allocate(n,d,false),kp=allocate(n,d,false),vp=allocate(n,d,false);
        pack_kernel<<<blocks(n*d),256,0,stream_>>>(q.data(),k?k->data():nullptr,v?v->data():nullptr,static_cast<float*>(qp.data()),static_cast<float*>(kp.data()),static_cast<float*>(vp.data()),n,d,heads,q.bf16,fused,fused?float(std::pow(double(hd),-.25)):1/std::sqrt(float(hd)));launch_check();
        constexpr size_t max_tile=128;
        for(size_t base=0;base<n;base+=max_tile){size_t tile=std::min(max_tile,n-base);auto scores=allocate(heads*tile,n,false),output=allocate(heads*tile,hd,false);float alpha=1,beta=0;
            check_blas(cublasSgemmStridedBatched(blas_,CUBLAS_OP_T,CUBLAS_OP_N,int(n),int(tile),int(hd),&alpha,static_cast<float*>(kp.data()),int(hd),n*hd,static_cast<float*>(qp.data())+base*hd,int(hd),n*hd,&beta,static_cast<float*>(scores.data()),int(n),tile*n,int(heads)));
            softmax_kernel<<<unsigned(heads*tile),256,0,stream_>>>(static_cast<float*>(scores.data()),n,tile,base,block,valid);launch_check();alpha=1;
            check_blas(cublasSgemmStridedBatched(blas_,CUBLAS_OP_N,CUBLAS_OP_N,int(hd),int(tile),int(n),&alpha,static_cast<float*>(vp.data()),int(hd),n*hd,static_cast<float*>(scores.data()),int(n),tile*n,&beta,static_cast<float*>(output.data()),int(hd),tile*hd,int(heads)));
            unpack_kernel<<<blocks(tile*d),256,0,stream_>>>(static_cast<float*>(output.data()),y.data(),n,d,heads,tile,base,q.bf16);launch_check();
        }return y;
    }
    Value pool(const Value&x,size_t f) override {auto y=allocate((x.rows+f-1)/f,x.cols,x.bf16);pool_kernel<<<blocks(y.size()),256,0,stream_>>>(x.data(),y.data(),y.size(),x.rows,x.cols,f,x.bf16);launch_check();return y;}
    Value interpolate(const Value&x,size_t s,size_t h,size_t w) override {auto y=allocate(h*w,x.cols,x.bf16);position_kernel<<<blocks(y.size()),256,0,stream_>>>(x.data(),y.data(),s,h,w,x.cols,x.bf16);launch_check();return y;}
    Value normalize(const Value&x) override {auto y=allocate(x.rows,x.cols,x.bf16);normalize_kernel<<<unsigned(x.rows),256,0,stream_>>>(x.data(),y.data(),x.cols,x.bf16);launch_check();return y;}
    Value squares(const Value&x) override {auto y=allocate(x.rows,1,x.bf16);squares_kernel<<<unsigned(x.rows),256,0,stream_>>>(x.data(),y.data(),x.cols,x.bf16);launch_check();return y;}
    std::vector<uint16_t> quantize(const Value&x,const Value&cb,const Value&cn,std::atomic<bool>&cancel) override {
        auto xn=squares(x),best=allocate(x.rows,1,false),ids=allocate(x.rows,1,true);best_init<<<blocks(x.rows),256,0,stream_>>>(static_cast<float*>(best.data()),static_cast<uint16_t*>(ids.data()),x.rows);launch_check();
        for(size_t base=0;base<cb.rows;base+=1024){if(cancel.load())throw Error(OMNI_CANCELLED,"Inference cancelled");size_t tile=std::min(size_t(1024),cb.rows-base);char*ptr=static_cast<char*>(cb.data())+base*cb.cols*(cb.bf16?2:4);Value part{std::shared_ptr<void>(cb.storage,ptr),tile,cb.cols,cb.bf16};auto dots=linear(x,part,nullptr);argmin_kernel<<<blocks(x.rows),256,0,stream_>>>(dots.data(),xn.data(),cn.data(),static_cast<float*>(best.data()),static_cast<uint16_t*>(ids.data()),x.rows,tile,base,x.bf16);launch_check();}
        std::vector<uint16_t> out(x.rows);check_cuda(cudaMemcpyAsync(out.data(),ids.data(),out.size()*2,cudaMemcpyDeviceToHost,stream_));synchronize();return out;
    }
    void synchronize() override {check_cuda(cudaStreamSynchronize(stream_));}
};
}
std::unique_ptr<Ops> make_cuda(int device){return std::make_unique<Cuda>(device);}
}
