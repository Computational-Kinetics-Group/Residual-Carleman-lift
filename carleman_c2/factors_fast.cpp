#include <cmath>
#include <omp.h>
extern "C" void evolve_fast(int N,int cap,int active,double rate,const double* b,const double* w,const double* center,const double* in,double* out,double* qout,double* pop,int threads){
int sites=N*N;
#pragma omp parallel for num_threads(threads) schedule(static)
for(int site=0;site<sites;site++){
const double* a0=in+((long)site*6+0)*cap;
const double* a1=in+((long)site*6+1)*cap;
const double* a2=in+((long)site*6+2)*cap;
const double* a3=in+((long)site*6+3)*cap;
const double* a4=in+((long)site*6+4)*cap;
const double* a5=in+((long)site*6+5)*cap;
double* p0=pop+((long)site*9+0)*cap;
double* p1=pop+((long)site*9+1)*cap;
double* p2=pop+((long)site*9+2)*cap;
double* p3=pop+((long)site*9+3)*cap;
double* p4=pop+((long)site*9+4)*cap;
double* p5=pop+((long)site*9+5)*cap;
double* p6=pop+((long)site*9+6)*cap;
double* p7=pop+((long)site*9+7)*cap;
double* p8=pop+((long)site*9+8)*cap;
#pragma omp simd
for(int j=0;j<active;j++){
double v0=a0[j];
double v1=a1[j];
double v2=a2[j];
double v3=a3[j];
double v4=a4[j];
double v5=a5[j];
p0[j]=b[0]*v0+b[1]*v1+b[2]*v2+b[3]*v3+b[4]*v4+b[5]*v5;
p1[j]=b[6]*v0+b[7]*v1+b[8]*v2+b[9]*v3+b[10]*v4+b[11]*v5;
p2[j]=b[12]*v0+b[13]*v1+b[14]*v2+b[15]*v3+b[16]*v4+b[17]*v5;
p3[j]=b[18]*v0+b[19]*v1+b[20]*v2+b[21]*v3+b[22]*v4+b[23]*v5;
p4[j]=b[24]*v0+b[25]*v1+b[26]*v2+b[27]*v3+b[28]*v4+b[29]*v5;
p5[j]=b[30]*v0+b[31]*v1+b[32]*v2+b[33]*v3+b[34]*v4+b[35]*v5;
p6[j]=b[36]*v0+b[37]*v1+b[38]*v2+b[39]*v3+b[40]*v4+b[41]*v5;
p7[j]=b[42]*v0+b[43]*v1+b[44]*v2+b[45]*v3+b[46]*v4+b[47]*v5;
p8[j]=b[48]*v0+b[49]*v1+b[50]*v2+b[51]*v3+b[52]*v4+b[53]*v5;
}}
#pragma omp parallel for num_threads(threads) schedule(static)
for(int site=0;site<sites;site++){
int x=site/N,y=site%N; const double* sc=center+site*6;
double ux=sc[1]/sc[0],uy=sc[2]/sc[0],om=1-rate;
const double* p0=pop+((long)(((x-(-1)+N)%N)*N+(y-(-1)+N)%N)*9+0)*cap;
const double* p1=pop+((long)(((x-(-1)+N)%N)*N+(y-(0)+N)%N)*9+1)*cap;
const double* p2=pop+((long)(((x-(-1)+N)%N)*N+(y-(1)+N)%N)*9+2)*cap;
const double* p3=pop+((long)(((x-(0)+N)%N)*N+(y-(-1)+N)%N)*9+3)*cap;
const double* p4=pop+((long)(((x-(0)+N)%N)*N+(y-(0)+N)%N)*9+4)*cap;
const double* p5=pop+((long)(((x-(0)+N)%N)*N+(y-(1)+N)%N)*9+5)*cap;
const double* p6=pop+((long)(((x-(1)+N)%N)*N+(y-(-1)+N)%N)*9+6)*cap;
const double* p7=pop+((long)(((x-(1)+N)%N)*N+(y-(0)+N)%N)*9+7)*cap;
const double* p8=pop+((long)(((x-(1)+N)%N)*N+(y-(1)+N)%N)*9+8)*cap;
double* o0=out+((long)site*6+0)*cap;
double* o1=out+((long)site*6+1)*cap;
double* o2=out+((long)site*6+2)*cap;
double* o3=out+((long)site*6+3)*cap;
double* o4=out+((long)site*6+4)*cap;
double* o5=out+((long)site*6+5)*cap;
double M[54],X[9],Y[9];
for(int i=0;i<9;i++){
 for(int l=0;l<3;l++)M[l*9+i]=w[i]*b[i*6+l];
 M[27+i]=w[i]*(rate*b[6*i+3]+om/std::sqrt(2.)*(2*ux*b[6*i+1]-ux*ux*b[6*i]));
 M[36+i]=w[i]*(rate*b[6*i+4]+om*(uy*b[6*i+1]+ux*b[6*i+2]-ux*uy*b[6*i]));
 M[45+i]=w[i]*(rate*b[6*i+5]+om/std::sqrt(2.)*(2*uy*b[6*i+2]-uy*uy*b[6*i]));
 X[i]=w[i]*(b[6*i+1]-ux*b[6*i]);Y[i]=w[i]*(b[6*i+2]-uy*b[6*i]);
}
auto apply=[&](int j,double &qx,double &qy) __attribute__((always_inline)) {
double v0=p0[j];
double v1=p1[j];
double v2=p2[j];
double v3=p3[j];
double v4=p4[j];
double v5=p5[j];
double v6=p6[j];
double v7=p7[j];
double v8=p8[j];
o0[j]=M[0]*v0+M[1]*v1+M[2]*v2+M[3]*v3+M[4]*v4+M[5]*v5+M[6]*v6+M[7]*v7+M[8]*v8;
o1[j]=M[9]*v0+M[10]*v1+M[11]*v2+M[12]*v3+M[13]*v4+M[14]*v5+M[15]*v6+M[16]*v7+M[17]*v8;
o2[j]=M[18]*v0+M[19]*v1+M[20]*v2+M[21]*v3+M[22]*v4+M[23]*v5+M[24]*v6+M[25]*v7+M[26]*v8;
o3[j]=M[27]*v0+M[28]*v1+M[29]*v2+M[30]*v3+M[31]*v4+M[32]*v5+M[33]*v6+M[34]*v7+M[35]*v8;
o4[j]=M[36]*v0+M[37]*v1+M[38]*v2+M[39]*v3+M[40]*v4+M[41]*v5+M[42]*v6+M[43]*v7+M[44]*v8;
o5[j]=M[45]*v0+M[46]*v1+M[47]*v2+M[48]*v3+M[49]*v4+M[50]*v5+M[51]*v6+M[52]*v7+M[53]*v8;
qx=X[0]*v0+X[1]*v1+X[2]*v2+X[3]*v3+X[4]*v4+X[5]*v5+X[6]*v6+X[7]*v7+X[8]*v8;
qy=Y[0]*v0+Y[1]*v1+Y[2]*v2+Y[3]*v3+Y[4]*v4+Y[5]*v5+Y[6]*v6+Y[7]*v7+Y[8]*v8;
};
double x0,y0;apply(0,x0,y0);double xx=x0*x0,xy=x0*y0,yy=y0*y0;
#pragma omp simd reduction(+:xx,xy,yy)
for(int j=1;j<active;j+=2){
 double xd,yd,xw,yw;apply(j,xd,yd);apply(j+1,xw,yw);
 xx+=2*xd*xw;xy+=xd*yw+xw*yd;yy+=2*yd*yw;
}
qout[site*6]=qout[site*6+1]=qout[site*6+2]=0;
qout[site*6+3]=om*xx/(sc[0]*std::sqrt(2.));qout[site*6+4]=om*xy/sc[0];qout[site*6+5]=om*yy/(sc[0]*std::sqrt(2.));
}}
