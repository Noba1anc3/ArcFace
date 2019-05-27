NV_GPU=$1 nvidia-docker run -it -p $2:$2 -v $(pwd):/workspace iva_base:v0.1
