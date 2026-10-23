#include "omni/omni.h"
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <cstdio>

static void check(bool condition){if(!condition){std::fprintf(stderr,"Native smoke failure: %s\n",omni_last_error());std::exit(1);}}
int main(){
    check(std::strcmp(omni_version(),"0.1.0")==0);
    omni_model *m=nullptr;
    check(omni_model_load("missing-model.otm",OMNI_CPU,0,&m)==OMNI_IO_ERROR);
    check(m==nullptr&&std::strlen(omni_last_error())>0);
    check(omni_model_load(nullptr,OMNI_CPU,0,&m)==OMNI_INVALID_ARGUMENT);
    check(omni_context_create(nullptr,nullptr)==OMNI_INVALID_ARGUMENT);
    check(omni_encode_audio(nullptr,nullptr,0,0,nullptr,0,nullptr)==OMNI_INVALID_ARGUMENT);
    check(omni_encode_image(nullptr,nullptr,0,0,nullptr,0,nullptr)==OMNI_INVALID_ARGUMENT);
    check(omni_context_peak_workspace(nullptr)==0);
    omni_model_free(nullptr);omni_context_free(nullptr);omni_context_cancel(nullptr);
    return 0;
}
