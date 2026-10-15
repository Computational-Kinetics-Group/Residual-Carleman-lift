#include <cmath>
#include <vector>
#include <algorithm>
#include <omp.h>
extern "C" void evolve(int N,int cap,int active,double rate,const double* basis,const double* weights,const double* center,const double* in,double* out,double* qout,int threads){
 const int sites=N*N;
 #pragma omp parallel num_threads(threads)
 {
 std::vector<double> buf(6*active),pop(active);
 #pragma omp for schedule(static)
 for(int site=0;site<sites;site++){
  std::fill(buf.begin(),buf.end(),0.0);
  int x=site/N,y=site%N;
  for(int i=0;i<9;i++){
   int cx=i/3-1,cy=i%3-1;
   int src=((x-cx+N)%N)*N+(y-cy+N)%N;
   std::fill(pop.begin(),pop.end(),0.0);
   for(int l=0;l<6;l++){
    const double* a=in+((long)src*6+l)*cap;double b=basis[i*6+l];
    #pragma omp simd
    for(int j=0;j<active;j++)pop[j]+=b*a[j];
   }
   for(int l=0;l<6;l++){
    double* a=buf.data()+l*active;double b=weights[i]*basis[i*6+l];
    #pragma omp simd
    for(int j=0;j<active;j++)a[j]+=b*pop[j];
   }
  }
  const double* s=center+site*6;double ux=s[1]/s[0],uy=s[2]/s[0],om=1-rate;
  double xx=0,xy=0,yy=0;
  double x0=buf[active]-ux*buf[0],y0=buf[2*active]-uy*buf[0];
  xx=x0*x0;xy=x0*y0;yy=y0*y0;
  for(int j=1;j<active;j+=2){
   double xd=buf[active+j]-ux*buf[j],yd=buf[2*active+j]-uy*buf[j];
   double xw=buf[active+j+1]-ux*buf[j+1],yw=buf[2*active+j+1]-uy*buf[j+1];
   xx+=2*xd*xw;xy+=xd*yw+xw*yd;yy+=2*yd*yw;
  }
  qout[site*6+0]=qout[site*6+1]=qout[site*6+2]=0;
  qout[site*6+3]=om*xx/(s[0]*std::sqrt(2.));qout[site*6+4]=om*xy/s[0];qout[site*6+5]=om*yy/(s[0]*std::sqrt(2.));
  for(int l=0;l<3;l++){
   double* dest=out+((long)site*6+l)*cap;double* b=buf.data()+l*active;
   #pragma omp simd
   for(int j=0;j<active;j++)dest[j]=b[j];
  }
  double* a3=out+((long)site*6+3)*cap;double* a4=out+((long)site*6+4)*cap;double* a5=out+((long)site*6+5)*cap;
  #pragma omp simd
  for(int j=0;j<active;j++){
   double r=buf[j],dx=buf[active+j],dy=buf[2*active+j];
   a3[j]=rate*buf[3*active+j]+om/std::sqrt(2.)*(2*ux*dx-ux*ux*r);
   a4[j]=rate*buf[4*active+j]+om*(uy*dx+ux*dy-ux*uy*r);
   a5[j]=rate*buf[5*active+j]+om/std::sqrt(2.)*(2*uy*dy-uy*uy*r);
  }
 }
 }
}
