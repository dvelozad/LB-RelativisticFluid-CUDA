#include<iostream>
#include<fstream>
#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
using namespace std;

#define M 20
#define N 10
#define W 10

#define Q 19

#define BLOCKSIZE_x 5
#define BLOCKSIZE_y 5
#define BLOCKSIZE_z 5


const int Mx=(M + BLOCKSIZE_x -1)/BLOCKSIZE_x;
const int My=(N + BLOCKSIZE_y -1)/BLOCKSIZE_y;
const int Mz=(W + BLOCKSIZE_z -1)/BLOCKSIZE_z;

const int cl = 1;
const int C = 1;
const float tau = 7.5;

__constant__ float d_w[Q];
__constant__ int d_Vx[Q];
__constant__ int d_Vy[Q];
__constant__ int d_Vz[Q];

__constant__ int d_cl = cl;
__constant__ int d_C = C;
__constant__ float d_tau = tau;

__constant__ int d_cm_x = 100;
__constant__ int d_cm_y = 50;
__constant__ int d_cm_z = 50;

__constant__ int d_R = 10;


/******************/
/* iDivUp FUNCTION */
/*******************/
int iDivUp(int a, int b){ return ((a % b) != 0) ? (a / b + 1) : (a / b); }

//-----------------------------------------------------------------------
//----------------------------Macroscopic Constrains---------------------
//-----------------------------------------------------------------------

//____________________________________Pressure__________________________________
__device__ float d_P(float g0,float g1,float g2,float g3,float g4,
                     float g5,float g6,float g7,float g8,float g9,
                     float g10,float g11,float g12,float g13,float g14,
                     float g15,float g16,float g17,float g18){
  int i,j; float sum1=0, sum2=0;
  float g_aux[19] = {g0,g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11,g12,g13,g14,g15,g16,g17,g18};
  for(i=0;i<Q;i++){
    sum1 += g_aux[i];
    for(j=0;j<Q;j++){
      sum2 += (g_aux[i]*g_aux[j]*(d_Vx[i]*d_Vx[j]+d_Vy[i]*d_Vy[j]+d_Vz[i]*d_Vz[j]));
    }
  }
  return -(1./3.)*sum1 + (1./3.)*sqrt(-3.*sum2 + 4.*sum1*sum1);
}
//_________________________________Energy Density______________________________
__device__ float d_rho(float g0,float g1,float g2,float g3,float g4,
                       float g5,float g6,float g7,float g8,float g9,
                       float g10,float g11,float g12,float g13,float g14,
                       float g15,float g16,float g17,float g18){
  return 3.*d_P(g0,g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11,g12,g13,g14,g15,g16,g17,g18);
}
//__________________________________Velocity Field______________________________
__device__ float d_Ux(float g0,float g1,float g2,float g3,float g4,
                      float g5,float g6,float g7,float g8,float g9,
                      float g10,float g11,float g12,float g13,float g14,
                      float g15,float g16,float g17,float g18){
  
  float sum1=0, sum2=0;
  
  sum2 = g0*d_Vx[0]+g1*d_Vx[1]+g2*d_Vx[2]+g3*d_Vx[3]+g4*d_Vx[4]+g5*d_Vx[5]+g6*d_Vx[6]+g7*d_Vx[7]+g8*d_Vx[8]+g9*d_Vx[9]+g10*d_Vx[10]+g11*d_Vx[11]+g12*d_Vx[12]+g13*d_Vx[13]+g14*d_Vx[14]+g15*d_Vx[15]+g16*d_Vx[16]+g17*d_Vx[17]+g18*d_Vx[18];

  sum1 = g0+g1+g2+g3+g4+g5+g6+g7+g8+g9+g10+g11+g12+g13+g14+g15+g16+g17+g18;
  
  return 3.*sum2/(3.*sum1 + 3.*d_P(g0,g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11,g12,g13,g14,g15,g16,g17,g18));
}
//--------------//
__device__ float d_Uy(float g0,float g1,float g2,float g3,float g4,
                      float g5,float g6,float g7,float g8,float g9,
                      float g10,float g11,float g12,float g13,float g14,
                      float g15,float g16,float g17,float g18){
  float sum1=0, sum2=0;

  sum2 = g0*d_Vy[0]+g1*d_Vy[1]+g2*d_Vy[2]+g3*d_Vy[3]+g4*d_Vy[4]+g5*d_Vy[5]+g6*d_Vy[6]+g7*d_Vy[7]+g8*d_Vy[8]+g9*d_Vy[9]+g10*d_Vy[10]+g11*d_Vy[11]+g12*d_Vy[12]+g13*d_Vy[13]+g14*d_Vy[14]+g15*d_Vy[15]+g16*d_Vy[16]+g17*d_Vy[17]+g18*d_Vy[18];

  sum1 = g0+g1+g2+g3+g4+g5+g6+g7+g8+g9+g10+g11+g12+g13+g14+g15+g16+g17+g18;
  
  return 3.*sum2/(3.*sum1 + 3.*d_P(g0,g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11,g12,g13,g14,g15,g16,g17,g18));
}
//-------------//
__device__ float d_Uz(float g0,float g1,float g2,float g3,float g4,
                      float g5,float g6,float g7,float g8,float g9,
                      float g10,float g11,float g12,float g13,float g14,
                      float g15,float g16,float g17,float g18){
  float sum1=0, sum2=0;

  sum2 = g0*d_Vz[0]+g1*d_Vz[1]+g2*d_Vz[2]+g3*d_Vz[3]+g4*d_Vz[4]+g5*d_Vz[5]+g6*d_Vz[6]+g7*d_Vz[7]+g8*d_Vz[8]+g9*d_Vz[9]+g10*d_Vz[10]+g11*d_Vz[11]+g12*d_Vz[12]+g13*d_Vz[13]+g14*d_Vz[14]+g15*d_Vz[15]+g16*d_Vz[16]+g17*d_Vz[17]+g18*d_Vz[18];

  sum1 = g0+g1+g2+g3+g4+g5+g6+g7+g8+g9+g10+g11+g12+g13+g14+g15+g16+g17+g18;
  
  return 3.*sum2/(3.*sum1 + 3.*d_P(g0,g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11,g12,g13,g14,g15,g16,g17,g18));
}
//___________________________________Gamma___________________________________
__device__ float d_gamma(float Ux0,float Uy0,float Uz0){
  float U2;
  U2 = Ux0*Ux0 + Uy0*Uy0 + Uz0*Uz0;
  return 1./sqrt(1.-(U2/(d_C*d_C)));
}
//______________________________Particle density______________________________
__device__ float d_n(float f0,float f1,float f2,float f3,float f4,
                     float f5,float f6,float f7,float f8,float f9,
                     float f10,float f11,float f12,float f13,float f14,
                     float f15,float f16,float f17,float f18,
                     float Ux0,float Uy0,float Uz0){
  float sum = 0;
  sum = f0+f1+f2+f3+f4+f5+f6+f7+f8+f9+f10+f11+f12+f13+f14+f15+f16+f17+f18;
  return sum/d_gamma(Ux0,Uy0,Uz0);
}
//------------------------------------------------------------------------------
//--------------------------------Equilibrium Functions-------------------------
//------------------------------------------------------------------------------

__device__ float d_feq(int i,float n0,float Ux0,float Uy0,float Uz0){
  float y,U2,UdotV;

  y = d_gamma(Ux0,Uy0,Uz0);
  UdotV = Ux0*d_Vx[i]+Uy0*d_Vy[i]+Uz0*d_Vz[i];
  U2 = Ux0*Ux0 + Uy0*Uy0 + Uz0*Uz0;

  return d_w[i]*n0*y*(1.+3.*UdotV/(d_cl*d_cl) + (9./2.)*(UdotV*UdotV)/(d_cl*d_cl*d_cl*d_cl) - (3./2.)*(U2/(d_cl*d_cl)));
}
__device__ float d_geq(int i,float rho0,float P0,float Ux0,float Uy0,float Uz0){
  float y2,UdotV,U2;

  y2 = d_gamma(Ux0,Uy0,Uz0)*d_gamma(Ux0,Uy0,Uz0);
  UdotV = Ux0*d_Vx[i]+Uy0*d_Vy[i]+Uz0*d_Vz[i];
  U2 = Ux0*Ux0 + Uy0*Uy0 + Uz0*Uz0;
  
  if(i == 0){
    return 3.*P0*d_w[0]*y2*(4. - (2.+ d_cl*d_cl)/(y2*d_cl*d_cl) - 2.*(U2/(d_cl*d_cl)));
  }else{
    return 3.*d_w[i]*P0*y2*( 1./(y2*d_cl*d_cl) + 4.*UdotV/(d_cl*d_cl) + 6.*(UdotV*UdotV)/(d_cl*d_cl*d_cl*d_cl) - 2.*(U2/(d_cl*d_cl)) );
  }
}
/**********************************************/
__global__ void op_indv_advection(cudaPitchedPtr devPitchedPtrI,cudaPitchedPtr devPitchedPtrInew,int I)
{  
  //printf("\o/");
  int ix =  blockIdx.x*blockDim.x+threadIdx.x;
  int iy =  blockIdx.y*blockDim.y+threadIdx.y;
  int iz =  blockIdx.z*blockDim.z+threadIdx.z;

  //--------------------------------------------
  int _ix = (M + ix + d_Vx[I])%M;
  int _iy = (N + iy + d_Vy[I])%N;
  int _iz = (W + iz + d_Vz[I])%W;
  
  char* devPtrI = (char*) devPitchedPtrI.ptr;
  size_t pitchI = devPitchedPtrI.pitch;
  size_t slicePitchI = pitchI * N;
  
  char* sliceI = devPtrI + _iz * slicePitchI;
  float* fI = (float*)(sliceI + _iy * pitchI);
  
  char* devPtrInew = (char*) devPitchedPtrInew.ptr;
  size_t pitchInew = devPitchedPtrInew.pitch;
  size_t slicePitchInew = pitchInew * N;
  
  char* sliceInew = devPtrInew + iz * slicePitchInew;
  float* fInew = (float*)(sliceInew + iy * pitchInew);
  
  //printf("| %i ",_iz);
  //if((_ix != M & _ix != -1) & (_iy != N & _iy != -1) & (_iz != W & _iz != -1)){
  if((ix >= 1 & ix < M-1) & (iy >= 1 & iy < N-1)  & (iz >= 1 & iz < W-1) ){
    fI[_ix] = fInew[ix];
  }
  //}
}
__global__ void d_collition(cudaPitchedPtr devpitchf0,cudaPitchedPtr devpitchf0new,cudaPitchedPtr devpitchg0,cudaPitchedPtr devpitchg0new,
                            cudaPitchedPtr devpitchf1,cudaPitchedPtr devpitchf1new,cudaPitchedPtr devpitchg1,cudaPitchedPtr devpitchg1new,
                            cudaPitchedPtr devpitchf2,cudaPitchedPtr devpitchf2new,cudaPitchedPtr devpitchg2,cudaPitchedPtr devpitchg2new,
                            cudaPitchedPtr devpitchf3,cudaPitchedPtr devpitchf3new,cudaPitchedPtr devpitchg3,cudaPitchedPtr devpitchg3new,
                            cudaPitchedPtr devpitchf4,cudaPitchedPtr devpitchf4new,cudaPitchedPtr devpitchg4,cudaPitchedPtr devpitchg4new,
                            cudaPitchedPtr devpitchf5,cudaPitchedPtr devpitchf5new,cudaPitchedPtr devpitchg5,cudaPitchedPtr devpitchg5new,
                            cudaPitchedPtr devpitchf6,cudaPitchedPtr devpitchf6new,cudaPitchedPtr devpitchg6,cudaPitchedPtr devpitchg6new,
                            cudaPitchedPtr devpitchf7,cudaPitchedPtr devpitchf7new,cudaPitchedPtr devpitchg7,cudaPitchedPtr devpitchg7new,
                            cudaPitchedPtr devpitchf8,cudaPitchedPtr devpitchf8new,cudaPitchedPtr devpitchg8,cudaPitchedPtr devpitchg8new,
                            cudaPitchedPtr devpitchf9,cudaPitchedPtr devpitchf9new,cudaPitchedPtr devpitchg9,cudaPitchedPtr devpitchg9new,
                            cudaPitchedPtr devpitchf10,cudaPitchedPtr devpitchf10new,cudaPitchedPtr devpitchg10,cudaPitchedPtr devpitchg10new,
                            cudaPitchedPtr devpitchf11,cudaPitchedPtr devpitchf11new,cudaPitchedPtr devpitchg11,cudaPitchedPtr devpitchg11new,
                            cudaPitchedPtr devpitchf12,cudaPitchedPtr devpitchf12new,cudaPitchedPtr devpitchg12,cudaPitchedPtr devpitchg12new,
                            cudaPitchedPtr devpitchf13,cudaPitchedPtr devpitchf13new,cudaPitchedPtr devpitchg13,cudaPitchedPtr devpitchg13new,
                            cudaPitchedPtr devpitchf14,cudaPitchedPtr devpitchf14new,cudaPitchedPtr devpitchg14,cudaPitchedPtr devpitchg14new,
                            cudaPitchedPtr devpitchf15,cudaPitchedPtr devpitchf15new,cudaPitchedPtr devpitchg15,cudaPitchedPtr devpitchg15new,
                            cudaPitchedPtr devpitchf16,cudaPitchedPtr devpitchf16new,cudaPitchedPtr devpitchg16,cudaPitchedPtr devpitchg16new,
                            cudaPitchedPtr devpitchf17,cudaPitchedPtr devpitchf17new,cudaPitchedPtr devpitchg17,cudaPitchedPtr devpitchg17new,
                            cudaPitchedPtr devpitchf18,cudaPitchedPtr devpitchf18new,cudaPitchedPtr devpitchg18,cudaPitchedPtr devpitchg18new){
  
  int ix =  blockIdx.x*blockDim.x+threadIdx.x;
  int iy =  blockIdx.y*blockDim.y+threadIdx.y;
  int iz =  blockIdx.z*blockDim.z+threadIdx.z;
  //printf("|%i",ix);
  //--------------------------------------------
  //--------------------------------------------
  
  char* devPtrf0 = (char*) devpitchf0.ptr;
  size_t pitchf0 = devpitchf0.pitch;
  size_t slicePitchf0 = pitchf0 * N;
  
  char* slicef0 = devPtrf0 + iz * slicePitchf0;
  float* f0 = (float*)(slicef0 + iy * pitchf0);
  
  char* devPtrf0new = (char*) devpitchf0new.ptr;
  size_t pitchf0new = devpitchf0new.pitch;
  size_t slicePitchf0new = pitchf0new * N;
  
  char* slicef0new = devPtrf0new + iz * slicePitchf0new;
  float* f0new = (float*)(slicef0new + iy * pitchf0new);
  //---------------------------------------------------
  char* devPtrg0 = (char*) devpitchg0.ptr;
  size_t pitchg0 = devpitchg0.pitch;
  size_t slicePitchg0 = pitchg0 * N;
  
  char* sliceg0 = devPtrg0 + iz * slicePitchg0;
  float* g0 = (float*)(sliceg0 + iy * pitchg0);
  
  char* devPtrg0new = (char*) devpitchg0new.ptr;
  size_t pitchg0new = devpitchg0new.pitch;
  size_t slicePitchg0new = pitchg0new * N;
  
  char* sliceg0new = devPtrg0new + iz * slicePitchg0new;
  float* g0new = (float*)(sliceg0new + iy * pitchg0new);
  //-------------------------------------------------
  //--------------------------------------------------
  char* devPtrf1 = (char*) devpitchf1.ptr;
  size_t pitchf1 = devpitchf1.pitch;
  size_t slicePitchf1 = pitchf1 * N;
  
  char* slicef1 = devPtrf1 + iz * slicePitchf1;
  float* f1 = (float*)(slicef1 + iy * pitchf1);
  
  char* devPtrf1new = (char*) devpitchf1new.ptr;
  size_t pitchf1new = devpitchf1new.pitch;
  size_t slicePitchf1new = pitchf1new * N;
  
  char* slicef1new = devPtrf1new + iz * slicePitchf1new;
  float* f1new = (float*)(slicef1new + iy * pitchf1new);
  //---------------------------------------------------
  char* devPtrg1 = (char*) devpitchg1.ptr;
  size_t pitchg1 = devpitchg1.pitch;
  size_t slicePitchg1 = pitchg1 * N;
  
  char* sliceg1 = devPtrg1 + iz * slicePitchg1;
  float* g1 = (float*)(sliceg1 + iy * pitchg1);
  
  char* devPtrg1new = (char*) devpitchg1new.ptr;
  size_t pitchg1new = devpitchg1new.pitch;
  size_t slicePitchg1new = pitchg1new * N;
  
  char* sliceg1new = devPtrg1new + iz * slicePitchg1new;
  float* g1new = (float*)(sliceg1new + iy * pitchg1new);
  //-------------------------------------------------
  //--------------------------------------------------
  char* devPtrf2 = (char*) devpitchf2.ptr;
  size_t pitchf2 = devpitchf2.pitch;
  size_t slicePitchf2 = pitchf2 * N;
  
  char* slicef2 = devPtrf2 + iz * slicePitchf2;
  float* f2 = (float*)(slicef2 + iy * pitchf2);
  
  char* devPtrf2new = (char*) devpitchf2new.ptr;
  size_t pitchf2new = devpitchf2new.pitch;
  size_t slicePitchf2new = pitchf2new * N;
  
  char* slicef2new = devPtrf2new + iz * slicePitchf2new;
  float* f2new = (float*)(slicef2new + iy * pitchf2new);
  //---------------------------------------------------
  char* devPtrg2 = (char*) devpitchg2.ptr;
  size_t pitchg2 = devpitchg2.pitch;
  size_t slicePitchg2 = pitchg2 * N;
  
  char* sliceg2 = devPtrg2 + iz * slicePitchg2;
  float* g2 = (float*)(sliceg2 + iy * pitchg2);
  
  char* devPtrg2new = (char*) devpitchg2new.ptr;
  size_t pitchg2new = devpitchg2new.pitch;
  size_t slicePitchg2new = pitchg2new * N;
  
  char* sliceg2new = devPtrg2new + iz * slicePitchg2new;
  float* g2new = (float*)(sliceg2new + iy * pitchg2new);
  //-------------------------------------------------
  //--------------------------------------------------
  char* devPtrf3 = (char*) devpitchf3.ptr;
  size_t pitchf3 = devpitchf3.pitch;
  size_t slicePitchf3 = pitchf3 * N;
  
  char* slicef3 = devPtrf3 + iz * slicePitchf3;
  float* f3 = (float*)(slicef3 + iy * pitchf3);
  
  char* devPtrf3new = (char*) devpitchf3new.ptr;
  size_t pitchf3new = devpitchf3new.pitch;
  size_t slicePitchf3new = pitchf3new * N;
  
  char* slicef3new = devPtrf3new + iz * slicePitchf3new;
  float* f3new = (float*)(slicef3new + iy * pitchf3new);
  //---------------------------------------------------
  char* devPtrg3 = (char*) devpitchg3.ptr;
  size_t pitchg3 = devpitchg3.pitch;
  size_t slicePitchg3 = pitchg3 * N;
  
  char* sliceg3 = devPtrg3 + iz * slicePitchg3;
  float* g3 = (float*)(sliceg3 + iy * pitchg3);
  
  char* devPtrg3new = (char*) devpitchg3new.ptr;
  size_t pitchg3new = devpitchg3new.pitch;
  size_t slicePitchg3new = pitchg3new * N;
  
  char* sliceg3new = devPtrg3new + iz * slicePitchg3new;
  float* g3new = (float*)(sliceg3new + iy * pitchg3new);
  //-------------------------------------------------
  //--------------------------------------------------
  char* devPtrf4 = (char*) devpitchf4.ptr;
  size_t pitchf4 = devpitchf4.pitch;
  size_t slicePitchf4 = pitchf4 * N;
  
  char* slicef4 = devPtrf4 + iz * slicePitchf4;
  float* f4 = (float*)(slicef4 + iy * pitchf4);
  
  char* devPtrf4new = (char*) devpitchf4new.ptr;
  size_t pitchf4new = devpitchf4new.pitch;
  size_t slicePitchf4new = pitchf4new * N;
  
  char* slicef4new = devPtrf4new + iz * slicePitchf4new;
  float* f4new = (float*)(slicef4new + iy * pitchf4new);
  //---------------------------------------------------
  char* devPtrg4 = (char*) devpitchg4.ptr;
  size_t pitchg4 = devpitchg4.pitch;
  size_t slicePitchg4 = pitchg4 * N;
  
  char* sliceg4 = devPtrg4 + iz * slicePitchg4;
  float* g4 = (float*)(sliceg4 + iy * pitchg4);
  
  char* devPtrg4new = (char*) devpitchg4new.ptr;
  size_t pitchg4new = devpitchg4new.pitch;
  size_t slicePitchg4new = pitchg4new * N;
  
  char* sliceg4new = devPtrg4new + iz * slicePitchg4new;
  float* g4new = (float*)(sliceg4new + iy * pitchg4new);
  //-------------------------------------------------
  //--------------------------------------------------
  char* devPtrf5 = (char*) devpitchf5.ptr;
  size_t pitchf5 = devpitchf5.pitch;
  size_t slicePitchf5 = pitchf5 * N;
  
  char* slicef5 = devPtrf5 + iz * slicePitchf5;
  float* f5 = (float*)(slicef5 + iy * pitchf5);
  
  char* devPtrf5new = (char*) devpitchf5new.ptr;
  size_t pitchf5new = devpitchf5new.pitch;
  size_t slicePitchf5new = pitchf5new * N;
  
  char* slicef5new = devPtrf5new + iz * slicePitchf5new;
  float* f5new = (float*)(slicef5new + iy * pitchf5new);
  //---------------------------------------------------
  char* devPtrg5 = (char*) devpitchg5.ptr;
  size_t pitchg5 = devpitchg5.pitch;
  size_t slicePitchg5 = pitchg5 * N;
  
  char* sliceg5 = devPtrg5 + iz * slicePitchg5;
  float* g5 = (float*)(sliceg5 + iy * pitchg5);
  
  char* devPtrg5new = (char*) devpitchg5new.ptr;
  size_t pitchg5new = devpitchg5new.pitch;
  size_t slicePitchg5new = pitchg5new * N;
  
  char* sliceg5new = devPtrg5new + iz * slicePitchg5new;
  float* g5new = (float*)(sliceg5new + iy * pitchg5new);
  //-------------------------------------------------
  //--------------------------------------------------
  char* devPtrf6 = (char*) devpitchf6.ptr;
  size_t pitchf6 = devpitchf6.pitch;
  size_t slicePitchf6 = pitchf6 * N;
  
  char* slicef6 = devPtrf6 + iz * slicePitchf6;
  float* f6 = (float*)(slicef6 + iy * pitchf6);
  
  char* devPtrf6new = (char*) devpitchf6new.ptr;
  size_t pitchf6new = devpitchf6new.pitch;
  size_t slicePitchf6new = pitchf6new * N;
  
  char* slicef6new = devPtrf6new + iz * slicePitchf6new;
  float* f6new = (float*)(slicef6new + iy * pitchf6new);
  //---------------------------------------------------
  char* devPtrg6 = (char*) devpitchg6.ptr;
  size_t pitchg6 = devpitchg6.pitch;
  size_t slicePitchg6 = pitchg6 * N;
  
  char* sliceg6 = devPtrg6 + iz * slicePitchg6;
  float* g6 = (float*)(sliceg6 + iy * pitchg6);
  
  char* devPtrg6new = (char*) devpitchg6new.ptr;
  size_t pitchg6new = devpitchg6new.pitch;
  size_t slicePitchg6new = pitchg6new * N;
  
  char* sliceg6new = devPtrg6new + iz * slicePitchg6new;
  float* g6new = (float*)(sliceg6new + iy * pitchg6new);
  //-------------------------------------------------
  //--------------------------------------------------
  char* devPtrf7 = (char*) devpitchf7.ptr;
  size_t pitchf7 = devpitchf7.pitch;
  size_t slicePitchf7 = pitchf7 * N;
  
  char* slicef7 = devPtrf7 + iz * slicePitchf7;
  float* f7 = (float*)(slicef7 + iy * pitchf7);
  
  char* devPtrf7new = (char*) devpitchf7new.ptr;
  size_t pitchf7new = devpitchf7new.pitch;
  size_t slicePitchf7new = pitchf7new * N;
  
  char* slicef7new = devPtrf7new + iz * slicePitchf7new;
  float* f7new = (float*)(slicef7new + iy * pitchf7new);
  //---------------------------------------------------
  char* devPtrg7 = (char*) devpitchg7.ptr;
  size_t pitchg7 = devpitchg7.pitch;
  size_t slicePitchg7 = pitchg7 * N;
  
  char* sliceg7 = devPtrg7 + iz * slicePitchg7;
  float* g7 = (float*)(sliceg7 + iy * pitchg7);
  
  char* devPtrg7new = (char*) devpitchg7new.ptr;
  size_t pitchg7new = devpitchg7new.pitch;
  size_t slicePitchg7new = pitchg7new * N;
  
  char* sliceg7new = devPtrg7new + iz * slicePitchg7new;
  float* g7new = (float*)(sliceg7new + iy * pitchg7new);
  //-------------------------------------------------
  //--------------------------------------------------
  char* devPtrf8 = (char*) devpitchf8.ptr;
  size_t pitchf8 = devpitchf8.pitch;
  size_t slicePitchf8 = pitchf8 * N;
  
  char* slicef8 = devPtrf8 + iz * slicePitchf8;
  float* f8 = (float*)(slicef8 + iy * pitchf8);
  
  char* devPtrf8new = (char*) devpitchf8new.ptr;
  size_t pitchf8new = devpitchf8new.pitch;
  size_t slicePitchf8new = pitchf8new * N;
  
  char* slicef8new = devPtrf8new + iz * slicePitchf8new;
  float* f8new = (float*)(slicef8new + iy * pitchf8new);
  //---------------------------------------------------
  char* devPtrg8 = (char*) devpitchg8.ptr;
  size_t pitchg8 = devpitchg8.pitch;
  size_t slicePitchg8 = pitchg8 * N;
  
  char* sliceg8 = devPtrg8 + iz * slicePitchg8;
  float* g8 = (float*)(sliceg8 + iy * pitchg8);
  
  char* devPtrg8new = (char*) devpitchg8new.ptr;
  size_t pitchg8new = devpitchg8new.pitch;
  size_t slicePitchg8new = pitchg8new * N;
  
  char* sliceg8new = devPtrg8new + iz * slicePitchg8new;
  float* g8new = (float*)(sliceg8new + iy * pitchg8new);
  //-------------------------------------------------
  //--------------------------------------------------
  char* devPtrf9 = (char*) devpitchf9.ptr;
  size_t pitchf9 = devpitchf9.pitch;
  size_t slicePitchf9 = pitchf9 * N;
  
  char* slicef9 = devPtrf9 + iz * slicePitchf9;
  float* f9 = (float*)(slicef9 + iy * pitchf9);
  
  char* devPtrf9new = (char*) devpitchf9new.ptr;
  size_t pitchf9new = devpitchf9new.pitch;
  size_t slicePitchf9new = pitchf9new * N;
  
  char* slicef9new = devPtrf9new + iz * slicePitchf9new;
  float* f9new = (float*)(slicef9new + iy * pitchf9new);
  //---------------------------------------------------
  char* devPtrg9 = (char*) devpitchg9.ptr;
  size_t pitchg9 = devpitchg9.pitch;
  size_t slicePitchg9 = pitchg9 * N;
  
  char* sliceg9 = devPtrg9 + iz * slicePitchg9;
  float* g9 = (float*)(sliceg9 + iy * pitchg9);
  
  char* devPtrg9new = (char*) devpitchg9new.ptr;
  size_t pitchg9new = devpitchg9new.pitch;
  size_t slicePitchg9new = pitchg9new * N;
  
  char* sliceg9new = devPtrg9new + iz * slicePitchg9new;
  float* g9new = (float*)(sliceg9new + iy * pitchg9new);
  //-------------------------------------------------
  //--------------------------------------------------
  char* devPtrf10 = (char*) devpitchf10.ptr;
  size_t pitchf10 = devpitchf10.pitch;
  size_t slicePitchf10 = pitchf10 * N;
  
  char* slicef10 = devPtrf10 + iz * slicePitchf10;
  float* f10 = (float*)(slicef10 + iy * pitchf10);
  
  char* devPtrf10new = (char*) devpitchf10new.ptr;
  size_t pitchf10new = devpitchf10new.pitch;
  size_t slicePitchf10new = pitchf10new * N;
  
  char* slicef10new = devPtrf10new + iz * slicePitchf10new;
  float* f10new = (float*)(slicef10new + iy * pitchf10new);
  //---------------------------------------------------
  char* devPtrg10 = (char*) devpitchg10.ptr;
  size_t pitchg10 = devpitchg10.pitch;
  size_t slicePitchg10 = pitchg10 * N;
  
  char* sliceg10 = devPtrg10 + iz * slicePitchg10;
  float* g10 = (float*)(sliceg10 + iy * pitchg10);
  
  char* devPtrg10new = (char*) devpitchg10new.ptr;
  size_t pitchg10new = devpitchg10new.pitch;
  size_t slicePitchg10new = pitchg10new * N;
  
  char* sliceg10new = devPtrg10new + iz * slicePitchg10new;
  float* g10new = (float*)(sliceg10new + iy * pitchg10new);
  //-------------------------------------------------
  //--------------------------------------------------
  char* devPtrf11 = (char*) devpitchf11.ptr;
  size_t pitchf11 = devpitchf11.pitch;
  size_t slicePitchf11 = pitchf11 * N;
  
  char* slicef11 = devPtrf11 + iz * slicePitchf11;
  float* f11 = (float*)(slicef11 + iy * pitchf11);
  
  char* devPtrf11new = (char*) devpitchf11new.ptr;
  size_t pitchf11new = devpitchf11new.pitch;
  size_t slicePitchf11new = pitchf11new * N;
  
  char* slicef11new = devPtrf11new + iz * slicePitchf11new;
  float* f11new = (float*)(slicef11new + iy * pitchf11new);
  //---------------------------------------------------
  char* devPtrg11 = (char*) devpitchg11.ptr;
  size_t pitchg11 = devpitchg11.pitch;
  size_t slicePitchg11 = pitchg11 * N;
  
  char* sliceg11 = devPtrg11 + iz * slicePitchg11;
  float* g11 = (float*)(sliceg11 + iy * pitchg11);
  
  char* devPtrg11new = (char*) devpitchg11new.ptr;
  size_t pitchg11new = devpitchg11new.pitch;
  size_t slicePitchg11new = pitchg11new * N;
  
  char* sliceg11new = devPtrg11new + iz * slicePitchg11new;
  float* g11new = (float*)(sliceg11new + iy * pitchg11new);
  //-------------------------------------------------
  //--------------------------------------------------
  char* devPtrf12 = (char*) devpitchf12.ptr;
  size_t pitchf12 = devpitchf12.pitch;
  size_t slicePitchf12 = pitchf12 * N;
  
  char* slicef12 = devPtrf12 + iz * slicePitchf12;
  float* f12 = (float*)(slicef12 + iy * pitchf12);
  
  char* devPtrf12new = (char*) devpitchf12new.ptr;
  size_t pitchf12new = devpitchf12new.pitch;
  size_t slicePitchf12new = pitchf12new * N;
  
  char* slicef12new = devPtrf12new + iz * slicePitchf12new;
  float* f12new = (float*)(slicef12new + iy * pitchf12new);
  //---------------------------------------------------
  char* devPtrg12 = (char*) devpitchg12.ptr;
  size_t pitchg12 = devpitchg12.pitch;
  size_t slicePitchg12 = pitchg12 * N;
  
  char* sliceg12 = devPtrg12 + iz * slicePitchg12;
  float* g12 = (float*)(sliceg12 + iy * pitchg12);
  
  char* devPtrg12new = (char*) devpitchg12new.ptr;
  size_t pitchg12new = devpitchg12new.pitch;
  size_t slicePitchg12new = pitchg12new * N;
  
  char* sliceg12new = devPtrg12new + iz * slicePitchg12new;
  float* g12new = (float*)(sliceg12new + iy * pitchg12new);
  //-------------------------------------------------
  //--------------------------------------------------
  char* devPtrf13 = (char*) devpitchf13.ptr;
  size_t pitchf13 = devpitchf13.pitch;
  size_t slicePitchf13 = pitchf13 * N;
  
  char* slicef13 = devPtrf13 + iz * slicePitchf13;
  float* f13 = (float*)(slicef13 + iy * pitchf13);
  
  char* devPtrf13new = (char*) devpitchf13new.ptr;
  size_t pitchf13new = devpitchf13new.pitch;
  size_t slicePitchf13new = pitchf13new * N;
  
  char* slicef13new = devPtrf13new + iz * slicePitchf13new;
  float* f13new = (float*)(slicef13new + iy * pitchf13new);
  //---------------------------------------------------
  char* devPtrg13 = (char*) devpitchg13.ptr;
  size_t pitchg13 = devpitchg13.pitch;
  size_t slicePitchg13 = pitchg13 * N;
  
  char* sliceg13 = devPtrg13 + iz * slicePitchg13;
  float* g13 = (float*)(sliceg13 + iy * pitchg13);
  
  char* devPtrg13new = (char*) devpitchg13new.ptr;
  size_t pitchg13new = devpitchg13new.pitch;
  size_t slicePitchg13new = pitchg13new * N;
  
  char* sliceg13new = devPtrg13new + iz * slicePitchg13new;
  float* g13new = (float*)(sliceg13new + iy * pitchg13new);
  //-------------------------------------------------
  //--------------------------------------------------
  char* devPtrf14 = (char*) devpitchf14.ptr;
  size_t pitchf14 = devpitchf14.pitch;
  size_t slicePitchf14 = pitchf14 * N;
  
  char* slicef14 = devPtrf14 + iz * slicePitchf14;
  float* f14 = (float*)(slicef14 + iy * pitchf14);
  
  char* devPtrf14new = (char*) devpitchf14new.ptr;
  size_t pitchf14new = devpitchf14new.pitch;
  size_t slicePitchf14new = pitchf14new * N;
  
  char* slicef14new = devPtrf14new + iz * slicePitchf14new;
  float* f14new = (float*)(slicef14new + iy * pitchf14new);
  //---------------------------------------------------
  char* devPtrg14 = (char*) devpitchg14.ptr;
  size_t pitchg14 = devpitchg14.pitch;
  size_t slicePitchg14 = pitchg14 * N;
  
  char* sliceg14 = devPtrg14 + iz * slicePitchg14;
  float* g14 = (float*)(sliceg14 + iy * pitchg14);
  
  char* devPtrg14new = (char*) devpitchg14new.ptr;
  size_t pitchg14new = devpitchg14new.pitch;
  size_t slicePitchg14new = pitchg14new * N;
  
  char* sliceg14new = devPtrg14new + iz * slicePitchg14new;
  float* g14new = (float*)(sliceg14new + iy * pitchg14new);
  //-------------------------------------------------
  //--------------------------------------------------
  char* devPtrf15 = (char*) devpitchf15.ptr;
  size_t pitchf15 = devpitchf15.pitch;
  size_t slicePitchf15 = pitchf15 * N;
  
  char* slicef15 = devPtrf15 + iz * slicePitchf15;
  float* f15 = (float*)(slicef15 + iy * pitchf15);
  
  char* devPtrf15new = (char*) devpitchf15new.ptr;
  size_t pitchf15new = devpitchf15new.pitch;
  size_t slicePitchf15new = pitchf15new * N;
  
  char* slicef15new = devPtrf15new + iz * slicePitchf15new;
  float* f15new = (float*)(slicef15new + iy * pitchf15new);
  //---------------------------------------------------
  char* devPtrg15 = (char*) devpitchg15.ptr;
  size_t pitchg15 = devpitchg15.pitch;
  size_t slicePitchg15 = pitchg15 * N;
  
  char* sliceg15 = devPtrg15 + iz * slicePitchg15;
  float* g15 = (float*)(sliceg15 + iy * pitchg15);
  
  char* devPtrg15new = (char*) devpitchg15new.ptr;
  size_t pitchg15new = devpitchg15new.pitch;
  size_t slicePitchg15new = pitchg15new * N;
  
  char* sliceg15new = devPtrg15new + iz * slicePitchg15new;
  float* g15new = (float*)(sliceg15new + iy * pitchg15new);
  //-------------------------------------------------
  //--------------------------------------------------
  char* devPtrf16 = (char*) devpitchf16.ptr;
  size_t pitchf16 = devpitchf16.pitch;
  size_t slicePitchf16 = pitchf16 * N;
  
  char* slicef16 = devPtrf16 + iz * slicePitchf16;
  float* f16 = (float*)(slicef16 + iy * pitchf16);
  
  char* devPtrf16new = (char*) devpitchf16new.ptr;
  size_t pitchf16new = devpitchf16new.pitch;
  size_t slicePitchf16new = pitchf16new * N;
  
  char* slicef16new = devPtrf16new + iz * slicePitchf16new;
  float* f16new = (float*)(slicef16new + iy * pitchf16new);
  //---------------------------------------------------
  char* devPtrg16 = (char*) devpitchg16.ptr;
  size_t pitchg16 = devpitchg16.pitch;
  size_t slicePitchg16 = pitchg16 * N;
  
  char* sliceg16 = devPtrg16 + iz * slicePitchg16;
  float* g16 = (float*)(sliceg16 + iy * pitchg16);
  
  char* devPtrg16new = (char*) devpitchg16new.ptr;
  size_t pitchg16new = devpitchg16new.pitch;
  size_t slicePitchg16new = pitchg16new * N;
  
  char* sliceg16new = devPtrg16new + iz * slicePitchg16new;
  float* g16new = (float*)(sliceg16new + iy * pitchg16new);
  //-------------------------------------------------
  //--------------------------------------------------
  char* devPtrf17 = (char*) devpitchf17.ptr;
  size_t pitchf17 = devpitchf17.pitch;
  size_t slicePitchf17 = pitchf17 * N;
  
  char* slicef17 = devPtrf17 + iz * slicePitchf17;
  float* f17 = (float*)(slicef17 + iy * pitchf17);
  
  char* devPtrf17new = (char*) devpitchf17new.ptr;
  size_t pitchf17new = devpitchf17new.pitch;
  size_t slicePitchf17new = pitchf17new * N;
  
  char* slicef17new = devPtrf17new + iz * slicePitchf17new;
  float* f17new = (float*)(slicef17new + iy * pitchf17new);
  //---------------------------------------------------
  char* devPtrg17 = (char*) devpitchg17.ptr;
  size_t pitchg17 = devpitchg17.pitch;
  size_t slicePitchg17 = pitchg17 * N;
  
  char* sliceg17 = devPtrg17 + iz * slicePitchg17;
  float* g17 = (float*)(sliceg17 + iy * pitchg17);
  
  char* devPtrg17new = (char*) devpitchg17new.ptr;
  size_t pitchg17new = devpitchg17new.pitch;
  size_t slicePitchg17new = pitchg17new * N;
  
  char* sliceg17new = devPtrg17new + iz * slicePitchg17new;
  float* g17new = (float*)(sliceg17new + iy * pitchg17new);
  //-------------------------------------------------
  //--------------------------------------------------
  char* devPtrf18 = (char*) devpitchf18.ptr;
  size_t pitchf18 = devpitchf18.pitch;
  size_t slicePitchf18 = pitchf18 * N;
  
  char* slicef18 = devPtrf18 + iz * slicePitchf18;
  float* f18 = (float*)(slicef18 + iy * pitchf18);
  
  char* devPtrf18new = (char*) devpitchf18new.ptr;
  size_t pitchf18new = devpitchf18new.pitch;
  size_t slicePitchf18new = pitchf18new * N;
  
  char* slicef18new = devPtrf18new + iz * slicePitchf18new;
  float* f18new = (float*)(slicef18new + iy * pitchf18new);
  //---------------------------------------------------
  char* devPtrg18 = (char*) devpitchg18.ptr;
  size_t pitchg18 = devpitchg18.pitch;
  size_t slicePitchg18 = pitchg18 * N;
  
  char* sliceg18 = devPtrg18 + iz * slicePitchg18;
  float* g18 = (float*)(sliceg18 + iy * pitchg18);
  
  char* devPtrg18new = (char*) devpitchg18new.ptr;
  size_t pitchg18new = devpitchg18new.pitch;
  size_t slicePitchg18new = pitchg18new * N;
  
  char* sliceg18new = devPtrg18new + iz * slicePitchg18new;
  float* g18new = (float*)(sliceg18new + iy * pitchg18new);
  //-------------------------------------------------
  //--------------------------------------------------
  //printf("|%f",f0[390]);
  float Ux0,Uy0,Uz0,n0,rho0,P0,T0;

  
  if( ((ix - d_cm_x)*(ix - d_cm_x) + (iy - d_cm_y)*(iy - d_cm_y) + (iz - d_cm_z)*(iz - d_cm_z)) <= d_R*d_R){
  Ux0 = 0; 
  Uy0 = 0;
  Uz0 = 0;

  T0 = 0.0314;

  P0 = 0;
  n0 = P0/T0;
  rho0 = 3*n0*T0;

  }else{

  Ux0 = d_Ux(g0[ix],g1[ix],g2[ix],g3[ix],g4[ix],g5[ix],g6[ix],g7[ix],g8[ix],g9[ix],g10[ix],g11[ix],g12[ix],g13[ix],g14[ix],g15[ix],g16[ix],g17[ix],g18[ix]);
  Uy0 = d_Uy(g0[ix],g1[ix],g2[ix],g3[ix],g4[ix],g5[ix],g6[ix],g7[ix],g8[ix],g9[ix],g10[ix],g11[ix],g12[ix],g13[ix],g14[ix],g15[ix],g16[ix],g17[ix],g18[ix]);
  Uz0 = d_Uz(g0[ix],g1[ix],g2[ix],g3[ix],g4[ix],g5[ix],g6[ix],g7[ix],g8[ix],g9[ix],g10[ix],g11[ix],g12[ix],g13[ix],g14[ix],g15[ix],g16[ix],g17[ix],g18[ix]);
  n0 = d_n(f0[ix],f1[ix],f2[ix],f3[ix],f4[ix],f5[ix],f6[ix],f7[ix],f8[ix],f9[ix],f10[ix],f11[ix],f12[ix],f13[ix],f14[ix],f15[ix],f16[ix],f17[ix],f18[ix],Ux0,Uy0,Uz0);
  rho0 = d_rho(g0[ix],g1[ix],g2[ix],g3[ix],g4[ix],g5[ix],g6[ix],g7[ix],g8[ix],g9[ix],g10[ix],g11[ix],g12[ix],g13[ix],g14[ix],g15[ix],g16[ix],g17[ix],g18[ix]);
  P0 = d_P(g0[ix],g1[ix],g2[ix],g3[ix],g4[ix],g5[ix],g6[ix],g7[ix],g8[ix],g9[ix],g10[ix],g11[ix],g12[ix],g13[ix],g14[ix],g15[ix],g16[ix],g17[ix],g18[ix]);
  }

  
  f0new[ix] = (1.-1./(d_tau))*f0[ix]+(1./d_tau)*d_feq(0,n0,Ux0,Uy0,Uz0);
  g0new[ix] = (1.-1./(d_tau))*g0[ix]+(1./d_tau)*d_geq(0,rho0,P0,Ux0,Uy0,Uz0);
  

  f1new[ix] = (1.-1./(d_tau))*f1[ix]+(1./d_tau)*d_feq(1,n0,Ux0,Uy0,Uz0);
  g1new[ix] = (1.-1./(d_tau))*g1[ix]+(1./d_tau)*d_geq(1,rho0,P0,Ux0,Uy0,Uz0);

  f2new[ix] = (1.-1./(d_tau))*f2[ix]+(1./d_tau)*d_feq(2,n0,Ux0,Uy0,Uz0);
  g2new[ix] = (1.-1./(d_tau))*g2[ix]+(1./d_tau)*d_geq(2,rho0,P0,Ux0,Uy0,Uz0);

  f3new[ix] = (1.-1./(d_tau))*f3[ix]+(1./d_tau)*d_feq(3,n0,Ux0,Uy0,Uz0);
  g3new[ix] = (1.-1./(d_tau))*g3[ix]+(1./d_tau)*d_geq(3,rho0,P0,Ux0,Uy0,Uz0);

  f4new[ix] = (1.-1./(d_tau))*f4[ix]+(1./d_tau)*d_feq(4,n0,Ux0,Uy0,Uz0);
  g4new[ix] = (1.-1./(d_tau))*g4[ix]+(1./d_tau)*d_geq(4,rho0,P0,Ux0,Uy0,Uz0);

  f5new[ix] = (1.-1./(d_tau))*f5[ix]+(1./d_tau)*d_feq(5,n0,Ux0,Uy0,Uz0);
  g5new[ix] = (1.-1./(d_tau))*g5[ix]+(1./d_tau)*d_geq(5,rho0,P0,Ux0,Uy0,Uz0);

  f6new[ix] = (1.-1./(d_tau))*f6[ix]+(1./d_tau)*d_feq(6,n0,Ux0,Uy0,Uz0);
  g6new[ix] = (1.-1./(d_tau))*g6[ix]+(1./d_tau)*d_geq(6,rho0,P0,Ux0,Uy0,Uz0);

  f7new[ix] = (1.-1./(d_tau))*f7[ix]+(1./d_tau)*d_feq(7,n0,Ux0,Uy0,Uz0);
  g7new[ix] = (1.-1./(d_tau))*g7[ix]+(1./d_tau)*d_geq(7,rho0,P0,Ux0,Uy0,Uz0);

  f8new[ix] = (1.-1./(d_tau))*f8[ix]+(1./d_tau)*d_feq(8,n0,Ux0,Uy0,Uz0);
  g8new[ix] = (1.-1./(d_tau))*g8[ix]+(1./d_tau)*d_geq(8,rho0,P0,Ux0,Uy0,Uz0);

  f9new[ix] = (1.-1./(d_tau))*f9[ix]+(1./d_tau)*d_feq(9,n0,Ux0,Uy0,Uz0);
  g9new[ix] = (1.-1./(d_tau))*g9[ix]+(1./d_tau)*d_geq(9,rho0,P0,Ux0,Uy0,Uz0);

  f10new[ix] = (1.-1./(d_tau))*f10[ix]+(1./d_tau)*d_feq(10,n0,Ux0,Uy0,Uz0);
  g10new[ix] = (1.-1./(d_tau))*g10[ix]+(1./d_tau)*d_geq(10,rho0,P0,Ux0,Uy0,Uz0);

  f11new[ix] = (1.-1./(d_tau))*f11[ix]+(1./d_tau)*d_feq(11,n0,Ux0,Uy0,Uz0);
  g11new[ix] = (1.-1./(d_tau))*g11[ix]+(1./d_tau)*d_geq(11,rho0,P0,Ux0,Uy0,Uz0);

  f12new[ix] = (1.-1./(d_tau))*f12[ix]+(1./d_tau)*d_feq(12,n0,Ux0,Uy0,Uz0);
  g12new[ix] = (1.-1./(d_tau))*g12[ix]+(1./d_tau)*d_geq(12,rho0,P0,Ux0,Uy0,Uz0);

  f13new[ix] = (1.-1./(d_tau))*f13[ix]+(1./d_tau)*d_feq(13,n0,Ux0,Uy0,Uz0);
  g13new[ix] = (1.-1./(d_tau))*g13[ix]+(1./d_tau)*d_geq(13,rho0,P0,Ux0,Uy0,Uz0);

  f14new[ix] = (1.-1./(d_tau))*f14[ix]+(1./d_tau)*d_feq(14,n0,Ux0,Uy0,Uz0);
  g14new[ix] = (1.-1./(d_tau))*g14[ix]+(1./d_tau)*d_geq(14,rho0,P0,Ux0,Uy0,Uz0);

  f15new[ix] = (1.-1./(d_tau))*f15[ix]+(1./d_tau)*d_feq(15,n0,Ux0,Uy0,Uz0);
  g15new[ix] = (1.-1./(d_tau))*g15[ix]+(1./d_tau)*d_geq(15,rho0,P0,Ux0,Uy0,Uz0);

  f16new[ix] = (1.-1./(d_tau))*f16[ix]+(1./d_tau)*d_feq(16,n0,Ux0,Uy0,Uz0);
  g16new[ix] = (1.-1./(d_tau))*g16[ix]+(1./d_tau)*d_geq(16,rho0,P0,Ux0,Uy0,Uz0);

  f17new[ix] = (1.-1./(d_tau))*f17[ix]+(1./d_tau)*d_feq(17,n0,Ux0,Uy0,Uz0);
  g17new[ix] = (1.-1./(d_tau))*g17[ix]+(1./d_tau)*d_geq(17,rho0,P0,Ux0,Uy0,Uz0);

  f18new[ix] = (1.-1./(d_tau))*f18[ix]+(1./d_tau)*d_feq(18,n0,Ux0,Uy0,Uz0);
  g18new[ix] = (1.-1./(d_tau))*g18[ix]+(1./d_tau)*d_geq(18,rho0,P0,Ux0,Uy0,Uz0); 

  /*if(ix == 400 || ix == 399){
	printf("|**%.9f, %i",g0new[ix],ix);
	printf("|**%.9f, %i",g1new[ix],ix);
	printf("|**%.9f, %i",g2new[ix],ix);
	printf("|**%.9f, %i",g3new[ix],ix);
	printf("|**%.9f, %i",g4new[ix],ix);
	printf("|**%.9f, %i",g5new[ix],ix);
	printf("|**%.9f, %i",g6new[ix],ix);
	printf("|**%.9f, %i",g7new[ix],ix);
  }*/
  

}
//--------------------Class-------------------------------
class LatticeBoltzmann{
private:
  float h_w[Q];
  int h_Vx[Q],h_Vy[Q],h_Vz[Q];
 
  
  cudaPitchedPtr devPitchedf0;           cudaPitchedPtr devPitchedf0new;           cudaPitchedPtr devPitchedg0;           cudaPitchedPtr devPitchedg0new;
  cudaPitchedPtr devPitchedf1;           cudaPitchedPtr devPitchedf1new;           cudaPitchedPtr devPitchedg1;           cudaPitchedPtr devPitchedg1new;
  cudaPitchedPtr devPitchedf2;           cudaPitchedPtr devPitchedf2new;           cudaPitchedPtr devPitchedg2;           cudaPitchedPtr devPitchedg2new;
  cudaPitchedPtr devPitchedf3;           cudaPitchedPtr devPitchedf3new;           cudaPitchedPtr devPitchedg3;           cudaPitchedPtr devPitchedg3new;
  cudaPitchedPtr devPitchedf4;           cudaPitchedPtr devPitchedf4new;           cudaPitchedPtr devPitchedg4;           cudaPitchedPtr devPitchedg4new;
  cudaPitchedPtr devPitchedf5;           cudaPitchedPtr devPitchedf5new;           cudaPitchedPtr devPitchedg5;           cudaPitchedPtr devPitchedg5new;
  cudaPitchedPtr devPitchedf6;           cudaPitchedPtr devPitchedf6new;           cudaPitchedPtr devPitchedg6;           cudaPitchedPtr devPitchedg6new;
  cudaPitchedPtr devPitchedf7;           cudaPitchedPtr devPitchedf7new;           cudaPitchedPtr devPitchedg7;           cudaPitchedPtr devPitchedg7new;
  cudaPitchedPtr devPitchedf8;           cudaPitchedPtr devPitchedf8new;           cudaPitchedPtr devPitchedg8;           cudaPitchedPtr devPitchedg8new;
  cudaPitchedPtr devPitchedf9;           cudaPitchedPtr devPitchedf9new;           cudaPitchedPtr devPitchedg9;           cudaPitchedPtr devPitchedg9new;
  cudaPitchedPtr devPitchedf10;           cudaPitchedPtr devPitchedf10new;           cudaPitchedPtr devPitchedg10;           cudaPitchedPtr devPitchedg10new;
  cudaPitchedPtr devPitchedf11;           cudaPitchedPtr devPitchedf11new;           cudaPitchedPtr devPitchedg11;           cudaPitchedPtr devPitchedg11new;
  cudaPitchedPtr devPitchedf12;           cudaPitchedPtr devPitchedf12new;           cudaPitchedPtr devPitchedg12;           cudaPitchedPtr devPitchedg12new;
  cudaPitchedPtr devPitchedf13;           cudaPitchedPtr devPitchedf13new;           cudaPitchedPtr devPitchedg13;           cudaPitchedPtr devPitchedg13new;
  cudaPitchedPtr devPitchedf14;           cudaPitchedPtr devPitchedf14new;           cudaPitchedPtr devPitchedg14;           cudaPitchedPtr devPitchedg14new;
  cudaPitchedPtr devPitchedf15;           cudaPitchedPtr devPitchedf15new;           cudaPitchedPtr devPitchedg15;           cudaPitchedPtr devPitchedg15new;
  cudaPitchedPtr devPitchedf16;           cudaPitchedPtr devPitchedf16new;           cudaPitchedPtr devPitchedg16;           cudaPitchedPtr devPitchedg16new;
  cudaPitchedPtr devPitchedf17;           cudaPitchedPtr devPitchedf17new;           cudaPitchedPtr devPitchedg17;           cudaPitchedPtr devPitchedg17new;
  cudaPitchedPtr devPitchedf18;           cudaPitchedPtr devPitchedf18new;           cudaPitchedPtr devPitchedg18;           cudaPitchedPtr devPitchedg18new;

  cudaMemcpy3DParms p0 = { 0 };          cudaMemcpy3DParms p0new = { 0 };            cudaMemcpy3DParms q0 = { 0 };          cudaMemcpy3DParms q0new = { 0 };
  cudaMemcpy3DParms p1 = { 0 };          cudaMemcpy3DParms p1new = { 0 };            cudaMemcpy3DParms q1 = { 0 };          cudaMemcpy3DParms q1new = { 0 };
  cudaMemcpy3DParms p2 = { 0 };          cudaMemcpy3DParms p2new = { 0 };            cudaMemcpy3DParms q2 = { 0 };          cudaMemcpy3DParms q2new = { 0 };
  cudaMemcpy3DParms p3 = { 0 };          cudaMemcpy3DParms p3new = { 0 };            cudaMemcpy3DParms q3 = { 0 };          cudaMemcpy3DParms q3new = { 0 };
  cudaMemcpy3DParms p4 = { 0 };          cudaMemcpy3DParms p4new = { 0 };            cudaMemcpy3DParms q4 = { 0 };          cudaMemcpy3DParms q4new = { 0 };
  cudaMemcpy3DParms p5 = { 0 };          cudaMemcpy3DParms p5new = { 0 };            cudaMemcpy3DParms q5 = { 0 };          cudaMemcpy3DParms q5new = { 0 };
  cudaMemcpy3DParms p6 = { 0 };          cudaMemcpy3DParms p6new = { 0 };            cudaMemcpy3DParms q6 = { 0 };          cudaMemcpy3DParms q6new = { 0 };
  cudaMemcpy3DParms p7 = { 0 };          cudaMemcpy3DParms p7new = { 0 };            cudaMemcpy3DParms q7 = { 0 };          cudaMemcpy3DParms q7new = { 0 };
  cudaMemcpy3DParms p8 = { 0 };          cudaMemcpy3DParms p8new = { 0 };            cudaMemcpy3DParms q8 = { 0 };          cudaMemcpy3DParms q8new = { 0 };
  cudaMemcpy3DParms p9 = { 0 };          cudaMemcpy3DParms p9new = { 0 };            cudaMemcpy3DParms q9 = { 0 };          cudaMemcpy3DParms q9new = { 0 };
  cudaMemcpy3DParms p10 = { 0 };          cudaMemcpy3DParms p10new = { 0 };            cudaMemcpy3DParms q10 = { 0 };          cudaMemcpy3DParms q10new = { 0 };
  cudaMemcpy3DParms p11 = { 0 };          cudaMemcpy3DParms p11new = { 0 };            cudaMemcpy3DParms q11 = { 0 };          cudaMemcpy3DParms q11new = { 0 };
  cudaMemcpy3DParms p12 = { 0 };          cudaMemcpy3DParms p12new = { 0 };            cudaMemcpy3DParms q12 = { 0 };          cudaMemcpy3DParms q12new = { 0 };
  cudaMemcpy3DParms p13 = { 0 };          cudaMemcpy3DParms p13new = { 0 };            cudaMemcpy3DParms q13 = { 0 };          cudaMemcpy3DParms q13new = { 0 };
  cudaMemcpy3DParms p14 = { 0 };          cudaMemcpy3DParms p14new = { 0 };            cudaMemcpy3DParms q14 = { 0 };          cudaMemcpy3DParms q14new = { 0 };
  cudaMemcpy3DParms p15 = { 0 };          cudaMemcpy3DParms p15new = { 0 };            cudaMemcpy3DParms q15 = { 0 };          cudaMemcpy3DParms q15new = { 0 };
  cudaMemcpy3DParms p16 = { 0 };          cudaMemcpy3DParms p16new = { 0 };            cudaMemcpy3DParms q16 = { 0 };          cudaMemcpy3DParms q16new = { 0 };
  cudaMemcpy3DParms p17 = { 0 };          cudaMemcpy3DParms p17new = { 0 };            cudaMemcpy3DParms q17 = { 0 };          cudaMemcpy3DParms q17new = { 0 };
  cudaMemcpy3DParms p18 = { 0 };          cudaMemcpy3DParms p18new = { 0 };            cudaMemcpy3DParms q18 = { 0 };          cudaMemcpy3DParms q18new = { 0 };
 
  float h_f0[W][N][M]; float h_f0new[W][N][M];  float h_g0[W][N][M];   float h_g0new[W][N][M];
  float h_f1[W][N][M]; float h_f1new[W][N][M];  float h_g1[W][N][M];   float h_g1new[W][N][M];
  float h_f2[W][N][M]; float h_f2new[W][N][M];  float h_g2[W][N][M];   float h_g2new[W][N][M];
  float h_f3[W][N][M]; float h_f3new[W][N][M];  float h_g3[W][N][M];   float h_g3new[W][N][M];
  float h_f4[W][N][M]; float h_f4new[W][N][M];  float h_g4[W][N][M];   float h_g4new[W][N][M];
  float h_f5[W][N][M]; float h_f5new[W][N][M];  float h_g5[W][N][M];   float h_g5new[W][N][M];
  float h_f6[W][N][M]; float h_f6new[W][N][M];  float h_g6[W][N][M];   float h_g6new[W][N][M];
  float h_f7[W][N][M]; float h_f7new[W][N][M];  float h_g7[W][N][M];   float h_g7new[W][N][M];
  float h_f8[W][N][M]; float h_f8new[W][N][M];  float h_g8[W][N][M];   float h_g8new[W][N][M];
  float h_f9[W][N][M]; float h_f9new[W][N][M];  float h_g9[W][N][M];   float h_g9new[W][N][M];
  float h_f10[W][N][M]; float h_f10new[W][N][M];  float h_g10[W][N][M];   float h_g10new[W][N][M];
  float h_f11[W][N][M]; float h_f11new[W][N][M];  float h_g11[W][N][M];   float h_g11new[W][N][M];
  float h_f12[W][N][M]; float h_f12new[W][N][M];  float h_g12[W][N][M];   float h_g12new[W][N][M];
  float h_f13[W][N][M]; float h_f13new[W][N][M];  float h_g13[W][N][M];   float h_g13new[W][N][M];
  float h_f14[W][N][M]; float h_f14new[W][N][M];  float h_g14[W][N][M];   float h_g14new[W][N][M];
  float h_f15[W][N][M]; float h_f15new[W][N][M];  float h_g15[W][N][M];   float h_g15new[W][N][M];
  float h_f16[W][N][M]; float h_f16new[W][N][M];  float h_g16[W][N][M];   float h_g16new[W][N][M];
  float h_f17[W][N][M]; float h_f17new[W][N][M];  float h_g17[W][N][M];   float h_g17new[W][N][M];
  float h_f18[W][N][M]; float h_f18new[W][N][M];  float h_g18[W][N][M];   float h_g18new[W][N][M];
  
public:
  LatticeBoltzmann(void);
  ~LatticeBoltzmann(void);
  void Start(float Ux0,float Uy0,float Uz0,float rho0, float rho1, float n0, float n1,float P0,float P1);
  void Advection(void);
  void Collision(void);
  void Show(void);
  float h_Ux(int ix,int iy,int iz);
  float h_Uy(int ix,int iy,int iz);
  float h_Uz(int ix,int iy,int iz);
  float h_gamma(float Ux0,float Uy0,float Uz0);
  float h_n(int ix,int iy,int iz,float Ux0,float Uy0,float Uz0);
  float h_P(int ix,int iy,int iz);
  float h_rho(int ix,int iy,int iz);
  float h_feq(int i,float n0,float Ux0,float Uy0,float Uz0);
  float h_geq(int i,float rho0,float P0,float Ux0,float Uy0,float Uz0);
  void Print(const char * NombreArchivo);
};

LatticeBoltzmann::LatticeBoltzmann(void){
  // --- 3D pitched allocation and host->device memcopy
  cudaExtent extent = make_cudaExtent(M * sizeof(float), N, W);
  cudaMalloc3D(&devPitchedf0, extent);   cudaMalloc3D(&devPitchedf0new, extent);   cudaMalloc3D(&devPitchedg0, extent);   cudaMalloc3D(&devPitchedg0new, extent);
  cudaMalloc3D(&devPitchedf1, extent);   cudaMalloc3D(&devPitchedf1new, extent);   cudaMalloc3D(&devPitchedg1, extent);   cudaMalloc3D(&devPitchedg1new, extent);
  cudaMalloc3D(&devPitchedf2, extent);   cudaMalloc3D(&devPitchedf2new, extent);   cudaMalloc3D(&devPitchedg2, extent);   cudaMalloc3D(&devPitchedg2new, extent);
  cudaMalloc3D(&devPitchedf3, extent);   cudaMalloc3D(&devPitchedf3new, extent);   cudaMalloc3D(&devPitchedg3, extent);   cudaMalloc3D(&devPitchedg3new, extent);
  cudaMalloc3D(&devPitchedf4, extent);   cudaMalloc3D(&devPitchedf4new, extent);   cudaMalloc3D(&devPitchedg4, extent);   cudaMalloc3D(&devPitchedg4new, extent);
  cudaMalloc3D(&devPitchedf5, extent);   cudaMalloc3D(&devPitchedf5new, extent);   cudaMalloc3D(&devPitchedg5, extent);   cudaMalloc3D(&devPitchedg5new, extent);
  cudaMalloc3D(&devPitchedf6, extent);   cudaMalloc3D(&devPitchedf6new, extent);   cudaMalloc3D(&devPitchedg6, extent);   cudaMalloc3D(&devPitchedg6new, extent);
  cudaMalloc3D(&devPitchedf7, extent);   cudaMalloc3D(&devPitchedf7new, extent);   cudaMalloc3D(&devPitchedg7, extent);   cudaMalloc3D(&devPitchedg7new, extent);
  cudaMalloc3D(&devPitchedf8, extent);   cudaMalloc3D(&devPitchedf8new, extent);   cudaMalloc3D(&devPitchedg8, extent);   cudaMalloc3D(&devPitchedg8new, extent);
  cudaMalloc3D(&devPitchedf9, extent);   cudaMalloc3D(&devPitchedf9new, extent);   cudaMalloc3D(&devPitchedg9, extent);   cudaMalloc3D(&devPitchedg9new, extent);
  cudaMalloc3D(&devPitchedf10, extent);   cudaMalloc3D(&devPitchedf10new, extent);   cudaMalloc3D(&devPitchedg10, extent);   cudaMalloc3D(&devPitchedg10new, extent);
  cudaMalloc3D(&devPitchedf11, extent);   cudaMalloc3D(&devPitchedf11new, extent);   cudaMalloc3D(&devPitchedg11, extent);   cudaMalloc3D(&devPitchedg11new, extent);
  cudaMalloc3D(&devPitchedf12, extent);   cudaMalloc3D(&devPitchedf12new, extent);   cudaMalloc3D(&devPitchedg12, extent);   cudaMalloc3D(&devPitchedg12new, extent);
  cudaMalloc3D(&devPitchedf13, extent);   cudaMalloc3D(&devPitchedf13new, extent);   cudaMalloc3D(&devPitchedg13, extent);   cudaMalloc3D(&devPitchedg13new, extent);
  cudaMalloc3D(&devPitchedf14, extent);   cudaMalloc3D(&devPitchedf14new, extent);   cudaMalloc3D(&devPitchedg14, extent);   cudaMalloc3D(&devPitchedg14new, extent);
  cudaMalloc3D(&devPitchedf15, extent);   cudaMalloc3D(&devPitchedf15new, extent);   cudaMalloc3D(&devPitchedg15, extent);   cudaMalloc3D(&devPitchedg15new, extent);
  cudaMalloc3D(&devPitchedf16, extent);   cudaMalloc3D(&devPitchedf16new, extent);   cudaMalloc3D(&devPitchedg16, extent);   cudaMalloc3D(&devPitchedg16new, extent);
  cudaMalloc3D(&devPitchedf17, extent);   cudaMalloc3D(&devPitchedf17new, extent);   cudaMalloc3D(&devPitchedg17, extent);   cudaMalloc3D(&devPitchedg17new, extent);
  cudaMalloc3D(&devPitchedf18, extent);   cudaMalloc3D(&devPitchedf18new, extent);   cudaMalloc3D(&devPitchedg18, extent);   cudaMalloc3D(&devPitchedg18new, extent);
}
LatticeBoltzmann::~LatticeBoltzmann(void){
  //Free memory on device
  cudaFree(&devPitchedf0.ptr);   cudaFree(&devPitchedf0new.ptr);   cudaFree(&devPitchedg0.ptr);   cudaFree(&devPitchedg0new.ptr);
  cudaFree(&devPitchedf1.ptr);   cudaFree(&devPitchedf1new.ptr);   cudaFree(&devPitchedg1.ptr);   cudaFree(&devPitchedg1new.ptr);
  cudaFree(&devPitchedf2.ptr);   cudaFree(&devPitchedf2new.ptr);   cudaFree(&devPitchedg2.ptr);   cudaFree(&devPitchedg2new.ptr);
  cudaFree(&devPitchedf3.ptr);   cudaFree(&devPitchedf3new.ptr);   cudaFree(&devPitchedg3.ptr);   cudaFree(&devPitchedg3new.ptr);
  cudaFree(&devPitchedf4.ptr);   cudaFree(&devPitchedf4new.ptr);   cudaFree(&devPitchedg4.ptr);   cudaFree(&devPitchedg4new.ptr);
  cudaFree(&devPitchedf5.ptr);   cudaFree(&devPitchedf5new.ptr);   cudaFree(&devPitchedg5.ptr);   cudaFree(&devPitchedg5new.ptr);
  cudaFree(&devPitchedf6.ptr);   cudaFree(&devPitchedf6new.ptr);   cudaFree(&devPitchedg6.ptr);   cudaFree(&devPitchedg6new.ptr);
  cudaFree(&devPitchedf7.ptr);   cudaFree(&devPitchedf7new.ptr);   cudaFree(&devPitchedg7.ptr);   cudaFree(&devPitchedg7new.ptr);
  cudaFree(&devPitchedf8.ptr);   cudaFree(&devPitchedf8new.ptr);   cudaFree(&devPitchedg8.ptr);   cudaFree(&devPitchedg8new.ptr);
  cudaFree(&devPitchedf9.ptr);   cudaFree(&devPitchedf9new.ptr);   cudaFree(&devPitchedg9.ptr);   cudaFree(&devPitchedg9new.ptr);
  cudaFree(&devPitchedf10.ptr);   cudaFree(&devPitchedf10new.ptr);   cudaFree(&devPitchedg10.ptr);   cudaFree(&devPitchedg10new.ptr);
  cudaFree(&devPitchedf11.ptr);   cudaFree(&devPitchedf11new.ptr);   cudaFree(&devPitchedg11.ptr);   cudaFree(&devPitchedg11new.ptr);
  cudaFree(&devPitchedf12.ptr);   cudaFree(&devPitchedf12new.ptr);   cudaFree(&devPitchedg12.ptr);   cudaFree(&devPitchedg12new.ptr);
  cudaFree(&devPitchedf13.ptr);   cudaFree(&devPitchedf13new.ptr);   cudaFree(&devPitchedg13.ptr);   cudaFree(&devPitchedg13new.ptr);
  cudaFree(&devPitchedf14.ptr);   cudaFree(&devPitchedf14new.ptr);   cudaFree(&devPitchedg14.ptr);   cudaFree(&devPitchedg14new.ptr);
  cudaFree(&devPitchedf15.ptr);   cudaFree(&devPitchedf15new.ptr);   cudaFree(&devPitchedg15.ptr);   cudaFree(&devPitchedg15new.ptr);
  cudaFree(&devPitchedf16.ptr);   cudaFree(&devPitchedf16new.ptr);   cudaFree(&devPitchedg16.ptr);   cudaFree(&devPitchedg16new.ptr);
  cudaFree(&devPitchedf17.ptr);   cudaFree(&devPitchedf17new.ptr);   cudaFree(&devPitchedg17.ptr);   cudaFree(&devPitchedg17new.ptr);
  cudaFree(&devPitchedf18.ptr);   cudaFree(&devPitchedf18new.ptr);   cudaFree(&devPitchedg18.ptr);   cudaFree(&devPitchedg18new.ptr);
}
void LatticeBoltzmann::Start(float Ux0,float Uy0,float Uz0,float rho0, float rho1, float n0, float n1,float P0,float P1){
  int i,j;
  int V[3][Q];
  //-----------Weights----------------
  h_w[0]=1./3.;
  for(i=1;i<7;i++)
    h_w[i]=1./18.;
  for(i=7;i<Q;i++)
    h_w[i]=1./36.;
  //-----------Velocities-------------
  for(i=0;i<Q;i++){
    for(j=0;j<3;j++){V[j][i] = 0;}
  }
  int counter = 0;
  for(i=1;i<7;i++){
    V[counter][i]=pow(-1,i+1);
    if(i%2==0){counter = counter+1;}
  }
  int counter1 = 0;
  int counter2 = 0;
  int counter3 = 0;
  for(i=7;i<Q;i++){
    if(i<15){
      V[counter3][i] = pow(-1,counter2);
      V[(counter1%2)+counter3+1][i] = pow(-1,(i+1)%2);
    }else{
      V[counter3][i] = pow(-1,counter1);
      V[counter3+1][i] = pow(-1,(i+1)%2);
    }
    if((i-6)%2==0){counter1 = counter1 + 1;}
    if((i-6)%4==0){counter2 = counter2 + 1;}
    if((i-6)%8==0){counter3 = counter3 + 1;}
  }
  for(i = 0; i < Q; i++){
    h_Vx[i] = V[0][i];
    h_Vy[i] = V[1][i];
    h_Vz[i] = V[2][i];
  }

  //------Enviarlas al Device-----------------
  cudaMemcpyToSymbol(d_w,h_w,Q*sizeof(float),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_Vx,h_Vx,Q*sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_Vy,h_Vy,Q*sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_Vz,h_Vz,Q*sizeof(int),0,cudaMemcpyHostToDevice);
  //FUNCIONES DE DISTRIBUCION
  int ix,iy,iz;
  float P,n,rho;
  //Cargar valores en el Host
  for(ix=0;ix<M;ix++)
    for(iy=0;iy<N;iy++)
      for(iz=0;iz<W;iz++){
        h_f0new[iz][iy][ix] = 0;    h_g0new[iz][iy][ix] = 0;
        h_f1new[iz][iy][ix] = 0;    h_g1new[iz][iy][ix] = 0;
        h_f2new[iz][iy][ix] = 0;    h_g2new[iz][iy][ix] = 0;
        h_f3new[iz][iy][ix] = 0;    h_g3new[iz][iy][ix] = 0;
        h_f4new[iz][iy][ix] = 0;    h_g4new[iz][iy][ix] = 0;
        h_f5new[iz][iy][ix] = 0;    h_g5new[iz][iy][ix] = 0;
        h_f6new[iz][iy][ix] = 0;    h_g6new[iz][iy][ix] = 0;
        h_f7new[iz][iy][ix] = 0;    h_g7new[iz][iy][ix] = 0;
        h_f8new[iz][iy][ix] = 0;    h_g8new[iz][iy][ix] = 0;
        h_f9new[iz][iy][ix] = 0;    h_g9new[iz][iy][ix] = 0;
        h_f10new[iz][iy][ix] =0;    h_g10new[iz][iy][ix] = 0;
        h_f11new[iz][iy][ix] =0;    h_g11new[iz][iy][ix] = 0;
        h_f12new[iz][iy][ix] =0;    h_g12new[iz][iy][ix] = 0;
        h_f13new[iz][iy][ix] =0;    h_g13new[iz][iy][ix] = 0;
        h_f14new[iz][iy][ix] =0;    h_g14new[iz][iy][ix] = 0;
        h_f15new[iz][iy][ix] =0;    h_g15new[iz][iy][ix] = 0;
        h_f16new[iz][iy][ix] =0;    h_g16new[iz][iy][ix] = 0;
        h_f17new[iz][iy][ix] =0;    h_g17new[iz][iy][ix] = 0;
        h_f18new[iz][iy][ix] =0;    h_g18new[iz][iy][ix] = 0;
	//---------------------------
        if(ix < int(M*0.5)){
          P = P0;
          n = n0;
          rho = rho0;
        }else if(ix >= int(M*0.5)){
          P = P1;
          n = n1;
          rho = rho1;
        }
	//--------------------------
        /*
	P = (P1-P0)*atan(ix-400)*0.5+(P0+P1)*0.5;
        n = (n1-n0)*atan(ix-400)*0.5+(n0+n1)*0.5;
        rho = (rho1-rho0)*atan(ix-400)*0.5+(rho0+rho1)*0.5;
        */

        h_f0[iz][iy][ix] = h_feq(0,n,Ux0,Uy0,Uz0);    h_g0[iz][iy][ix] = h_geq(0,rho,P,Ux0,Uy0,Uz0);
        h_f1[iz][iy][ix] = h_feq(1,n,Ux0,Uy0,Uz0);    h_g1[iz][iy][ix] = h_geq(1,rho,P,Ux0,Uy0,Uz0);
        h_f2[iz][iy][ix] = h_feq(2,n,Ux0,Uy0,Uz0);    h_g2[iz][iy][ix] = h_geq(2,rho,P,Ux0,Uy0,Uz0);
        h_f3[iz][iy][ix] = h_feq(3,n,Ux0,Uy0,Uz0);    h_g3[iz][iy][ix] = h_geq(3,rho,P,Ux0,Uy0,Uz0);
        h_f4[iz][iy][ix] = h_feq(4,n,Ux0,Uy0,Uz0);    h_g4[iz][iy][ix] = h_geq(4,rho,P,Ux0,Uy0,Uz0);
        h_f5[iz][iy][ix] = h_feq(5,n,Ux0,Uy0,Uz0);    h_g5[iz][iy][ix] = h_geq(5,rho,P,Ux0,Uy0,Uz0);
        h_f6[iz][iy][ix] = h_feq(6,n,Ux0,Uy0,Uz0);    h_g6[iz][iy][ix] = h_geq(6,rho,P,Ux0,Uy0,Uz0);
        h_f7[iz][iy][ix] = h_feq(7,n,Ux0,Uy0,Uz0);    h_g7[iz][iy][ix] = h_geq(7,rho,P,Ux0,Uy0,Uz0);
        h_f8[iz][iy][ix] = h_feq(8,n,Ux0,Uy0,Uz0);    h_g8[iz][iy][ix] = h_geq(8,rho,P,Ux0,Uy0,Uz0);
        h_f9[iz][iy][ix] = h_feq(9,n,Ux0,Uy0,Uz0);    h_g9[iz][iy][ix] = h_geq(9,rho,P,Ux0,Uy0,Uz0);
        h_f10[iz][iy][ix] = h_feq(10,n,Ux0,Uy0,Uz0);    h_g10[iz][iy][ix] = h_geq(10,rho,P,Ux0,Uy0,Uz0);
        h_f11[iz][iy][ix] = h_feq(11,n,Ux0,Uy0,Uz0);    h_g11[iz][iy][ix] = h_geq(11,rho,P,Ux0,Uy0,Uz0);
        h_f12[iz][iy][ix] = h_feq(12,n,Ux0,Uy0,Uz0);    h_g12[iz][iy][ix] = h_geq(12,rho,P,Ux0,Uy0,Uz0);
        h_f13[iz][iy][ix] = h_feq(13,n,Ux0,Uy0,Uz0);    h_g13[iz][iy][ix] = h_geq(13,rho,P,Ux0,Uy0,Uz0);
        h_f14[iz][iy][ix] = h_feq(14,n,Ux0,Uy0,Uz0);    h_g14[iz][iy][ix] = h_geq(14,rho,P,Ux0,Uy0,Uz0);
        h_f15[iz][iy][ix] = h_feq(15,n,Ux0,Uy0,Uz0);    h_g15[iz][iy][ix] = h_geq(15,rho,P,Ux0,Uy0,Uz0);
        h_f16[iz][iy][ix] = h_feq(16,n,Ux0,Uy0,Uz0);    h_g16[iz][iy][ix] = h_geq(16,rho,P,Ux0,Uy0,Uz0);
        h_f17[iz][iy][ix] = h_feq(17,n,Ux0,Uy0,Uz0);    h_g17[iz][iy][ix] = h_geq(17,rho,P,Ux0,Uy0,Uz0);
        h_f18[iz][iy][ix] = h_feq(18,n,Ux0,Uy0,Uz0);    h_g18[iz][iy][ix] = h_geq(18,rho,P,Ux0,Uy0,Uz0);
      }
  //cout << h_g10[0][0][39] << endl;
  //Llevar al Devic
  p0.srcPtr.ptr = h_f0;
  p0.srcPtr.pitch = M * sizeof(float);
  p0.srcPtr.xsize = M;
  p0.srcPtr.ysize = N;
  p0.dstPtr.ptr = devPitchedf0.ptr;
  p0.dstPtr.pitch = devPitchedf0.pitch;
  p0.dstPtr.xsize = M;
  p0.dstPtr.ysize = N;
  p0.extent.width = M * sizeof(float);
  p0.extent.height = N;
  p0.extent.depth = W;
  p0.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&p0);

  p1.srcPtr.ptr = h_f1;
  p1.srcPtr.pitch = M * sizeof(float);
  p1.srcPtr.xsize = M;
  p1.srcPtr.ysize = N;
  p1.dstPtr.ptr = devPitchedf1.ptr;
  p1.dstPtr.pitch = devPitchedf1.pitch;
  p1.dstPtr.xsize = M;
  p1.dstPtr.ysize = N;
  p1.extent.width = M * sizeof(float);
  p1.extent.height = N;
  p1.extent.depth = W;
  p1.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&p1);

  p2.srcPtr.ptr = h_f2;
  p2.srcPtr.pitch = M * sizeof(float);
  p2.srcPtr.xsize = M;
  p2.srcPtr.ysize = N;
  p2.dstPtr.ptr = devPitchedf2.ptr;
  p2.dstPtr.pitch = devPitchedf2.pitch;
  p2.dstPtr.xsize = M;
  p2.dstPtr.ysize = N;
  p2.extent.width = M * sizeof(float);
  p2.extent.height = N;
  p2.extent.depth = W;
  p2.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&p2);

  p3.srcPtr.ptr = h_f3;
  p3.srcPtr.pitch = M * sizeof(float);
  p3.srcPtr.xsize = M;
  p3.srcPtr.ysize = N;
  p3.dstPtr.ptr = devPitchedf3.ptr;
  p3.dstPtr.pitch = devPitchedf3.pitch;
  p3.dstPtr.xsize = M;
  p3.dstPtr.ysize = N;
  p3.extent.width = M * sizeof(float);
  p3.extent.height = N;
  p3.extent.depth = W;
  p3.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&p3);

  p4.srcPtr.ptr = h_f4;
  p4.srcPtr.pitch = M * sizeof(float);
  p4.srcPtr.xsize = M;
  p4.srcPtr.ysize = N;
  p4.dstPtr.ptr = devPitchedf4.ptr;
  p4.dstPtr.pitch = devPitchedf4.pitch;
  p4.dstPtr.xsize = M;
  p4.dstPtr.ysize = N;
  p4.extent.width = M * sizeof(float);
  p4.extent.height = N;
  p4.extent.depth = W;
  p4.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&p4);

  p5.srcPtr.ptr = h_f5;
  p5.srcPtr.pitch = M * sizeof(float);
  p5.srcPtr.xsize = M;
  p5.srcPtr.ysize = N;
  p5.dstPtr.ptr = devPitchedf5.ptr;
  p5.dstPtr.pitch = devPitchedf5.pitch;
  p5.dstPtr.xsize = M;
  p5.dstPtr.ysize = N;
  p5.extent.width = M * sizeof(float);
  p5.extent.height = N;
  p5.extent.depth = W;
  p5.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&p5);

  p6.srcPtr.ptr = h_f6;
  p6.srcPtr.pitch = M * sizeof(float);
  p6.srcPtr.xsize = M;
  p6.srcPtr.ysize = N;
  p6.dstPtr.ptr = devPitchedf6.ptr;
  p6.dstPtr.pitch = devPitchedf6.pitch;
  p6.dstPtr.xsize = M;
  p6.dstPtr.ysize = N;
  p6.extent.width = M * sizeof(float);
  p6.extent.height = N;
  p6.extent.depth = W;
  p6.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&p6);

  p7.srcPtr.ptr = h_f7;
  p7.srcPtr.pitch = M * sizeof(float);
  p7.srcPtr.xsize = M;
  p7.srcPtr.ysize = N;
  p7.dstPtr.ptr = devPitchedf7.ptr;
  p7.dstPtr.pitch = devPitchedf7.pitch;
  p7.dstPtr.xsize = M;
  p7.dstPtr.ysize = N;
  p7.extent.width = M * sizeof(float);
  p7.extent.height = N;
  p7.extent.depth = W;
  p7.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&p7);

  p8.srcPtr.ptr = h_f8;
  p8.srcPtr.pitch = M * sizeof(float);
  p8.srcPtr.xsize = M;
  p8.srcPtr.ysize = N;
  p8.dstPtr.ptr = devPitchedf8.ptr;
  p8.dstPtr.pitch = devPitchedf8.pitch;
  p8.dstPtr.xsize = M;
  p8.dstPtr.ysize = N;
  p8.extent.width = M * sizeof(float);
  p8.extent.height = N;
  p8.extent.depth = W;
  p8.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&p8);

  p9.srcPtr.ptr = h_f9;
  p9.srcPtr.pitch = M * sizeof(float);
  p9.srcPtr.xsize = M;
  p9.srcPtr.ysize = N;
  p9.dstPtr.ptr = devPitchedf9.ptr;
  p9.dstPtr.pitch = devPitchedf9.pitch;
  p9.dstPtr.xsize = M;
  p9.dstPtr.ysize = N;
  p9.extent.width = M * sizeof(float);
  p9.extent.height = N;
  p9.extent.depth = W;
  p9.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&p9);

  p10.srcPtr.ptr = h_f10;
  p10.srcPtr.pitch = M * sizeof(float);
  p10.srcPtr.xsize = M;
  p10.srcPtr.ysize = N;
  p10.dstPtr.ptr = devPitchedf10.ptr;
  p10.dstPtr.pitch = devPitchedf10.pitch;
  p10.dstPtr.xsize = M;
  p10.dstPtr.ysize = N;
  p10.extent.width = M * sizeof(float);
  p10.extent.height = N;
  p10.extent.depth = W;
  p10.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&p10);


  p11.srcPtr.ptr = h_f11;
  p11.srcPtr.pitch = M * sizeof(float);
  p11.srcPtr.xsize = M;
  p11.srcPtr.ysize = N;
  p11.dstPtr.ptr = devPitchedf11.ptr;
  p11.dstPtr.pitch = devPitchedf11.pitch;
  p11.dstPtr.xsize = M;
  p11.dstPtr.ysize = N;
  p11.extent.width = M * sizeof(float);
  p11.extent.height = N;
  p11.extent.depth = W;
  p11.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&p11);


  p12.srcPtr.ptr = h_f12;
  p12.srcPtr.pitch = M * sizeof(float);
  p12.srcPtr.xsize = M;
  p12.srcPtr.ysize = N;
  p12.dstPtr.ptr = devPitchedf12.ptr;
  p12.dstPtr.pitch = devPitchedf12.pitch;
  p12.dstPtr.xsize = M;
  p12.dstPtr.ysize = N;
  p12.extent.width = M * sizeof(float);
  p12.extent.height = N;
  p12.extent.depth = W;
  p12.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&p12);



  p13.srcPtr.ptr = h_f13;
  p13.srcPtr.pitch = M * sizeof(float);
  p13.srcPtr.xsize = M;
  p13.srcPtr.ysize = N;
  p13.dstPtr.ptr = devPitchedf13.ptr;
  p13.dstPtr.pitch = devPitchedf13.pitch;
  p13.dstPtr.xsize = M;
  p13.dstPtr.ysize = N;
  p13.extent.width = M * sizeof(float);
  p13.extent.height = N;
  p13.extent.depth = W;
  p13.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&p13);


  p14.srcPtr.ptr = h_f14;
  p14.srcPtr.pitch = M * sizeof(float);
  p14.srcPtr.xsize = M;
  p14.srcPtr.ysize = N;
  p14.dstPtr.ptr = devPitchedf14.ptr;
  p14.dstPtr.pitch = devPitchedf14.pitch;
  p14.dstPtr.xsize = M;
  p14.dstPtr.ysize = N;
  p14.extent.width = M * sizeof(float);
  p14.extent.height = N;
  p14.extent.depth = W;
  p14.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&p14);


  p15.srcPtr.ptr = h_f15;
  p15.srcPtr.pitch = M * sizeof(float);
  p15.srcPtr.xsize = M;
  p15.srcPtr.ysize = N;
  p15.dstPtr.ptr = devPitchedf15.ptr;
  p15.dstPtr.pitch = devPitchedf15.pitch;
  p15.dstPtr.xsize = M;
  p15.dstPtr.ysize = N;
  p15.extent.width = M * sizeof(float);
  p15.extent.height = N;
  p15.extent.depth = W;
  p15.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&p15);

  p16.srcPtr.ptr = h_f16;
  p16.srcPtr.pitch = M * sizeof(float);
  p16.srcPtr.xsize = M;
  p16.srcPtr.ysize = N;
  p16.dstPtr.ptr = devPitchedf16.ptr;
  p16.dstPtr.pitch = devPitchedf16.pitch;
  p16.dstPtr.xsize = M;
  p16.dstPtr.ysize = N;
  p16.extent.width = M * sizeof(float);
  p16.extent.height = N;
  p16.extent.depth = W;
  p16.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&p16);

  p17.srcPtr.ptr = h_f17;
  p17.srcPtr.pitch = M * sizeof(float);
  p17.srcPtr.xsize = M;
  p17.srcPtr.ysize = N;
  p17.dstPtr.ptr = devPitchedf17.ptr;
  p17.dstPtr.pitch = devPitchedf17.pitch;
  p17.dstPtr.xsize = M;
  p17.dstPtr.ysize = N;
  p17.extent.width = M * sizeof(float);
  p17.extent.height = N;
  p17.extent.depth = W;
  p17.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&p17);

  p18.srcPtr.ptr = h_f18;
  p18.srcPtr.pitch = M * sizeof(float);
  p18.srcPtr.xsize = M;
  p18.srcPtr.ysize = N;
  p18.dstPtr.ptr = devPitchedf18.ptr;
  p18.dstPtr.pitch = devPitchedf18.pitch;
  p18.dstPtr.xsize = M;
  p18.dstPtr.ysize = N;
  p18.extent.width = M * sizeof(float);
  p18.extent.height = N;
  p18.extent.depth = W;
  p18.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&p18);

  p0new.srcPtr.ptr = h_f0new;
  p0new.srcPtr.pitch = M * sizeof(float);
  p0new.srcPtr.xsize = M;
  p0new.srcPtr.ysize = N;
  p0new.dstPtr.ptr = devPitchedf0new.ptr;
  p0new.dstPtr.pitch = devPitchedf0new.pitch;
  p0new.dstPtr.xsize = M;
  p0new.dstPtr.ysize = N;
  p0new.extent.width = M * sizeof(float);
  p0new.extent.height = N;
  p0new.extent.depth = W;
  p0new.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&p0new);

  p1new.srcPtr.ptr = h_f1new;
  p1new.srcPtr.pitch = M * sizeof(float);
  p1new.srcPtr.xsize = M;
  p1new.srcPtr.ysize = N;
  p1new.dstPtr.ptr = devPitchedf1new.ptr;
  p1new.dstPtr.pitch = devPitchedf1new.pitch;
  p1new.dstPtr.xsize = M;
  p1new.dstPtr.ysize = N;
  p1new.extent.width = M * sizeof(float);
  p1new.extent.height = N;
  p1new.extent.depth = W;
  p1new.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&p1new);

  p2new.srcPtr.ptr = h_f2new;
  p2new.srcPtr.pitch = M * sizeof(float);
  p2new.srcPtr.xsize = M;
  p2new.srcPtr.ysize = N;
  p2new.dstPtr.ptr = devPitchedf2new.ptr;
  p2new.dstPtr.pitch = devPitchedf2new.pitch;
  p2new.dstPtr.xsize = M;
  p2new.dstPtr.ysize = N;
  p2new.extent.width = M * sizeof(float);
  p2new.extent.height = N;
  p2new.extent.depth = W;
  p2new.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&p2new);

  p3new.srcPtr.ptr = h_f3new;
  p3new.srcPtr.pitch = M * sizeof(float);
  p3new.srcPtr.xsize = M;
  p3new.srcPtr.ysize = N;
  p3new.dstPtr.ptr = devPitchedf3new.ptr;
  p3new.dstPtr.pitch = devPitchedf3new.pitch;
  p3new.dstPtr.xsize = M;
  p3new.dstPtr.ysize = N;
  p3new.extent.width = M * sizeof(float);
  p3new.extent.height = N;
  p3new.extent.depth = W;
  p3new.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&p3new);

  p4new.srcPtr.ptr = h_f4new;
  p4new.srcPtr.pitch = M * sizeof(float);
  p4new.srcPtr.xsize = M;
  p4new.srcPtr.ysize = N;
  p4new.dstPtr.ptr = devPitchedf4new.ptr;
  p4new.dstPtr.pitch = devPitchedf4new.pitch;
  p4new.dstPtr.xsize = M;
  p4new.dstPtr.ysize = N;
  p4new.extent.width = M * sizeof(float);
  p4new.extent.height = N;
  p4new.extent.depth = W;
  p4new.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&p4new);

  p5new.srcPtr.ptr = h_f5new;
  p5new.srcPtr.pitch = M * sizeof(float);
  p5new.srcPtr.xsize = M;
  p5new.srcPtr.ysize = N;
  p5new.dstPtr.ptr = devPitchedf5new.ptr;
  p5new.dstPtr.pitch = devPitchedf5new.pitch;
  p5new.dstPtr.xsize = M;
  p5new.dstPtr.ysize = N;
  p5new.extent.width = M * sizeof(float);
  p5new.extent.height = N;
  p5new.extent.depth = W;
  p5new.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&p5new);

  p6new.srcPtr.ptr = h_f6new;
  p6new.srcPtr.pitch = M * sizeof(float);
  p6new.srcPtr.xsize = M;
  p6new.srcPtr.ysize = N;
  p6new.dstPtr.ptr = devPitchedf6new.ptr;
  p6new.dstPtr.pitch = devPitchedf6new.pitch;
  p6new.dstPtr.xsize = M;
  p6new.dstPtr.ysize = N;
  p6new.extent.width = M * sizeof(float);
  p6new.extent.height = N;
  p6new.extent.depth = W;
  p6new.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&p6new);

  p7new.srcPtr.ptr = h_f7new;
  p7new.srcPtr.pitch = M * sizeof(float);
  p7new.srcPtr.xsize = M;
  p7new.srcPtr.ysize = N;
  p7new.dstPtr.ptr = devPitchedf7new.ptr;
  p7new.dstPtr.pitch = devPitchedf7new.pitch;
  p7new.dstPtr.xsize = M;
  p7new.dstPtr.ysize = N;
  p7new.extent.width = M * sizeof(float);
  p7new.extent.height = N;
  p7new.extent.depth = W;
  p7new.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&p7new);

  p8new.srcPtr.ptr = h_f8new;
  p8new.srcPtr.pitch = M * sizeof(float);
  p8new.srcPtr.xsize = M;
  p8new.srcPtr.ysize = N;
  p8new.dstPtr.ptr = devPitchedf8new.ptr;
  p8new.dstPtr.pitch = devPitchedf8new.pitch;
  p8new.dstPtr.xsize = M;
  p8new.dstPtr.ysize = N;
  p8new.extent.width = M * sizeof(float);
  p8new.extent.height = N;
  p8new.extent.depth = W;
  p8new.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&p8new);

  p9new.srcPtr.ptr = h_f9new;
  p9new.srcPtr.pitch = M * sizeof(float);
  p9new.srcPtr.xsize = M;
  p9new.srcPtr.ysize = N;
  p9new.dstPtr.ptr = devPitchedf9new.ptr;
  p9new.dstPtr.pitch = devPitchedf9new.pitch;
  p9new.dstPtr.xsize = M;
  p9new.dstPtr.ysize = N;
  p9new.extent.width = M * sizeof(float);
  p9new.extent.height = N;
  p9new.extent.depth = W;
  p9new.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&p9new);

  p10new.srcPtr.ptr = h_f10new;
  p10new.srcPtr.pitch = M * sizeof(float);
  p10new.srcPtr.xsize = M;
  p10new.srcPtr.ysize = N;
  p10new.dstPtr.ptr = devPitchedf10new.ptr;
  p10new.dstPtr.pitch = devPitchedf10new.pitch;
  p10new.dstPtr.xsize = M;
  p10new.dstPtr.ysize = N;
  p10new.extent.width = M * sizeof(float);
  p10new.extent.height = N;
  p10new.extent.depth = W;
  p10new.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&p10new);

  p11new.srcPtr.ptr = h_f11new;
  p11new.srcPtr.pitch = M * sizeof(float);
  p11new.srcPtr.xsize = M;
  p11new.srcPtr.ysize = N;
  p11new.dstPtr.ptr = devPitchedf11new.ptr;
  p11new.dstPtr.pitch = devPitchedf11new.pitch;
  p11new.dstPtr.xsize = M;
  p11new.dstPtr.ysize = N;
  p11new.extent.width = M * sizeof(float);
  p11new.extent.height = N;
  p11new.extent.depth = W;
  p11new.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&p11new);

  p12new.srcPtr.ptr = h_f12new;
  p12new.srcPtr.pitch = M * sizeof(float);
  p12new.srcPtr.xsize = M;
  p12new.srcPtr.ysize = N;
  p12new.dstPtr.ptr = devPitchedf12new.ptr;
  p12new.dstPtr.pitch = devPitchedf12new.pitch;
  p12new.dstPtr.xsize = M;
  p12new.dstPtr.ysize = N;
  p12new.extent.width = M * sizeof(float);
  p12new.extent.height = N;
  p12new.extent.depth = W;
  p12new.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&p12new);

  p13new.srcPtr.ptr = h_f13new;
  p13new.srcPtr.pitch = M * sizeof(float);
  p13new.srcPtr.xsize = M;
  p13new.srcPtr.ysize = N;
  p13new.dstPtr.ptr = devPitchedf13new.ptr;
  p13new.dstPtr.pitch = devPitchedf13new.pitch;
  p13new.dstPtr.xsize = M;
  p13new.dstPtr.ysize = N;
  p13new.extent.width = M * sizeof(float);
  p13new.extent.height = N;
  p13new.extent.depth = W;
  p13new.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&p13new);

  p14new.srcPtr.ptr = h_f14new;
  p14new.srcPtr.pitch = M * sizeof(float);
  p14new.srcPtr.xsize = M;
  p14new.srcPtr.ysize = N;
  p14new.dstPtr.ptr = devPitchedf14new.ptr;
  p14new.dstPtr.pitch = devPitchedf14new.pitch;
  p14new.dstPtr.xsize = M;
  p14new.dstPtr.ysize = N;
  p14new.extent.width = M * sizeof(float);
  p14new.extent.height = N;
  p14new.extent.depth = W;
  p14new.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&p14new);

  p15new.srcPtr.ptr = h_f15new;
  p15new.srcPtr.pitch = M * sizeof(float);
  p15new.srcPtr.xsize = M;
  p15new.srcPtr.ysize = N;
  p15new.dstPtr.ptr = devPitchedf15new.ptr;
  p15new.dstPtr.pitch = devPitchedf15new.pitch;
  p15new.dstPtr.xsize = M;
  p15new.dstPtr.ysize = N;
  p15new.extent.width = M * sizeof(float);
  p15new.extent.height = N;
  p15new.extent.depth = W;
  p15new.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&p15new);

  p16new.srcPtr.ptr = h_f16new;
  p16new.srcPtr.pitch = M * sizeof(float);
  p16new.srcPtr.xsize = M;
  p16new.srcPtr.ysize = N;
  p16new.dstPtr.ptr = devPitchedf16new.ptr;
  p16new.dstPtr.pitch = devPitchedf16new.pitch;
  p16new.dstPtr.xsize = M;
  p16new.dstPtr.ysize = N;
  p16new.extent.width = M * sizeof(float);
  p16new.extent.height = N;
  p16new.extent.depth = W;
  p16new.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&p16new);

  p17new.srcPtr.ptr = h_f17new;
  p17new.srcPtr.pitch = M * sizeof(float);
  p17new.srcPtr.xsize = M;
  p17new.srcPtr.ysize = N;
  p17new.dstPtr.ptr = devPitchedf17new.ptr;
  p17new.dstPtr.pitch = devPitchedf17new.pitch;
  p17new.dstPtr.xsize = M;
  p17new.dstPtr.ysize = N;
  p17new.extent.width = M * sizeof(float);
  p17new.extent.height = N;
  p17new.extent.depth = W;
  p17new.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&p17new);

  p18new.srcPtr.ptr = h_f18new;
  p18new.srcPtr.pitch = M * sizeof(float);
  p18new.srcPtr.xsize = M;
  p18new.srcPtr.ysize = N;
  p18new.dstPtr.ptr = devPitchedf18new.ptr;
  p18new.dstPtr.pitch = devPitchedf18new.pitch;
  p18new.dstPtr.xsize = M;
  p18new.dstPtr.ysize = N;
  p18new.extent.width = M * sizeof(float);
  p18new.extent.height = N;
  p18new.extent.depth = W;
  p18new.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&p18new);

  q0.srcPtr.ptr = h_g0;
  q0.srcPtr.pitch = M * sizeof(float);
  q0.srcPtr.xsize = M;
  q0.srcPtr.ysize = N;
  q0.dstPtr.ptr = devPitchedg0.ptr;
  q0.dstPtr.pitch = devPitchedg0.pitch;
  q0.dstPtr.xsize = M;
  q0.dstPtr.ysize = N;
  q0.extent.width = M * sizeof(float);
  q0.extent.height = N;
  q0.extent.depth = W;
  q0.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&q0);

  q1.srcPtr.ptr = h_g1;
  q1.srcPtr.pitch = M * sizeof(float);
  q1.srcPtr.xsize = M;
  q1.srcPtr.ysize = N;
  q1.dstPtr.ptr = devPitchedg1.ptr;
  q1.dstPtr.pitch = devPitchedg1.pitch;
  q1.dstPtr.xsize = M;
  q1.dstPtr.ysize = N;
  q1.extent.width = M * sizeof(float);
  q1.extent.height = N;
  q1.extent.depth = W;
  q1.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&q1);

  q2.srcPtr.ptr = h_g2;
  q2.srcPtr.pitch = M * sizeof(float);
  q2.srcPtr.xsize = M;
  q2.srcPtr.ysize = N;
  q2.dstPtr.ptr = devPitchedg2.ptr;
  q2.dstPtr.pitch = devPitchedg2.pitch;
  q2.dstPtr.xsize = M;
  q2.dstPtr.ysize = N;
  q2.extent.width = M * sizeof(float);
  q2.extent.height = N;
  q2.extent.depth = W;
  q2.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&q2);

  q3.srcPtr.ptr = h_g3;
  q3.srcPtr.pitch = M * sizeof(float);
  q3.srcPtr.xsize = M;
  q3.srcPtr.ysize = N;
  q3.dstPtr.ptr = devPitchedg3.ptr;
  q3.dstPtr.pitch = devPitchedg3.pitch;
  q3.dstPtr.xsize = M;
  q3.dstPtr.ysize = N;
  q3.extent.width = M * sizeof(float);
  q3.extent.height = N;
  q3.extent.depth = W;
  q3.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&q3);

  q4.srcPtr.ptr = h_g4;
  q4.srcPtr.pitch = M * sizeof(float);
  q4.srcPtr.xsize = M;
  q4.srcPtr.ysize = N;
  q4.dstPtr.ptr = devPitchedg4.ptr;
  q4.dstPtr.pitch = devPitchedg4.pitch;
  q4.dstPtr.xsize = M;
  q4.dstPtr.ysize = N;
  q4.extent.width = M * sizeof(float);
  q4.extent.height = N;
  q4.extent.depth = W;
  q4.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&q4);

  q5.srcPtr.ptr = h_g5;
  q5.srcPtr.pitch = M * sizeof(float);
  q5.srcPtr.xsize = M;
  q5.srcPtr.ysize = N;
  q5.dstPtr.ptr = devPitchedg5.ptr;
  q5.dstPtr.pitch = devPitchedg5.pitch;
  q5.dstPtr.xsize = M;
  q5.dstPtr.ysize = N;
  q5.extent.width = M * sizeof(float);
  q5.extent.height = N;
  q5.extent.depth = W;
  q5.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&q5);

  q6.srcPtr.ptr = h_g6;
  q6.srcPtr.pitch = M * sizeof(float);
  q6.srcPtr.xsize = M;
  q6.srcPtr.ysize = N;
  q6.dstPtr.ptr = devPitchedg6.ptr;
  q6.dstPtr.pitch = devPitchedg6.pitch;
  q6.dstPtr.xsize = M;
  q6.dstPtr.ysize = N;
  q6.extent.width = M * sizeof(float);
  q6.extent.height = N;
  q6.extent.depth = W;
  q6.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&q6);

  q7.srcPtr.ptr = h_g7;
  q7.srcPtr.pitch = M * sizeof(float);
  q7.srcPtr.xsize = M;
  q7.srcPtr.ysize = N;
  q7.dstPtr.ptr = devPitchedg7.ptr;
  q7.dstPtr.pitch = devPitchedg7.pitch;
  q7.dstPtr.xsize = M;
  q7.dstPtr.ysize = N;
  q7.extent.width = M * sizeof(float);
  q7.extent.height = N;
  q7.extent.depth = W;
  q7.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&q7);

  q8.srcPtr.ptr = h_g8;
  q8.srcPtr.pitch = M * sizeof(float);
  q8.srcPtr.xsize = M;
  q8.srcPtr.ysize = N;
  q8.dstPtr.ptr = devPitchedg8.ptr;
  q8.dstPtr.pitch = devPitchedg8.pitch;
  q8.dstPtr.xsize = M;
  q8.dstPtr.ysize = N;
  q8.extent.width = M * sizeof(float);
  q8.extent.height = N;
  q8.extent.depth = W;
  q8.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&q8);

  q9.srcPtr.ptr = h_g9;
  q9.srcPtr.pitch = M * sizeof(float);
  q9.srcPtr.xsize = M;
  q9.srcPtr.ysize = N;
  q9.dstPtr.ptr = devPitchedg9.ptr;
  q9.dstPtr.pitch = devPitchedg9.pitch;
  q9.dstPtr.xsize = M;
  q9.dstPtr.ysize = N;
  q9.extent.width = M * sizeof(float);
  q9.extent.height = N;
  q9.extent.depth = W;
  q9.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&q9);

  q10.srcPtr.ptr = h_g10;
  q10.srcPtr.pitch = M * sizeof(float);
  q10.srcPtr.xsize = M;
  q10.srcPtr.ysize = N;
  q10.dstPtr.ptr = devPitchedg10.ptr;
  q10.dstPtr.pitch = devPitchedg10.pitch;
  q10.dstPtr.xsize = M;
  q10.dstPtr.ysize = N;
  q10.extent.width = M * sizeof(float);
  q10.extent.height = N;
  q10.extent.depth = W;
  q10.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&q10);

  q11.srcPtr.ptr = h_g11;
  q11.srcPtr.pitch = M * sizeof(float);
  q11.srcPtr.xsize = M;
  q11.srcPtr.ysize = N;
  q11.dstPtr.ptr = devPitchedg11.ptr;
  q11.dstPtr.pitch = devPitchedg11.pitch;
  q11.dstPtr.xsize = M;
  q11.dstPtr.ysize = N;
  q11.extent.width = M * sizeof(float);
  q11.extent.height = N;
  q11.extent.depth = W;
  q11.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&q11);

  q12.srcPtr.ptr = h_g12;
  q12.srcPtr.pitch = M * sizeof(float);
  q12.srcPtr.xsize = M;
  q12.srcPtr.ysize = N;
  q12.dstPtr.ptr = devPitchedg12.ptr;
  q12.dstPtr.pitch = devPitchedg12.pitch;
  q12.dstPtr.xsize = M;
  q12.dstPtr.ysize = N;
  q12.extent.width = M * sizeof(float);
  q12.extent.height = N;
  q12.extent.depth = W;
  q12.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&q12);

  q13.srcPtr.ptr = h_g13;
  q13.srcPtr.pitch = M * sizeof(float);
  q13.srcPtr.xsize = M;
  q13.srcPtr.ysize = N;
  q13.dstPtr.ptr = devPitchedg13.ptr;
  q13.dstPtr.pitch = devPitchedg13.pitch;
  q13.dstPtr.xsize = M;
  q13.dstPtr.ysize = N;
  q13.extent.width = M * sizeof(float);
  q13.extent.height = N;
  q13.extent.depth = W;
  q13.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&q13);

  q14.srcPtr.ptr = h_g14;
  q14.srcPtr.pitch = M * sizeof(float);
  q14.srcPtr.xsize = M;
  q14.srcPtr.ysize = N;
  q14.dstPtr.ptr = devPitchedg14.ptr;
  q14.dstPtr.pitch = devPitchedg14.pitch;
  q14.dstPtr.xsize = M;
  q14.dstPtr.ysize = N;
  q14.extent.width = M * sizeof(float);
  q14.extent.height = N;
  q14.extent.depth = W;
  q14.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&q14);

  q15.srcPtr.ptr = h_g15;
  q15.srcPtr.pitch = M * sizeof(float);
  q15.srcPtr.xsize = M;
  q15.srcPtr.ysize = N;
  q15.dstPtr.ptr = devPitchedg15.ptr;
  q15.dstPtr.pitch = devPitchedg15.pitch;
  q15.dstPtr.xsize = M;
  q15.dstPtr.ysize = N;
  q15.extent.width = M * sizeof(float);
  q15.extent.height = N;
  q15.extent.depth = W;
  q15.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&q15);

  q16.srcPtr.ptr = h_g16;
  q16.srcPtr.pitch = M * sizeof(float);
  q16.srcPtr.xsize = M;
  q16.srcPtr.ysize = N;
  q16.dstPtr.ptr = devPitchedg16.ptr;
  q16.dstPtr.pitch = devPitchedg16.pitch;
  q16.dstPtr.xsize = M;
  q16.dstPtr.ysize = N;
  q16.extent.width = M * sizeof(float);
  q16.extent.height = N;
  q16.extent.depth = W;
  q16.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&q16);


  q17.srcPtr.ptr = h_g17;
  q17.srcPtr.pitch = M * sizeof(float);
  q17.srcPtr.xsize = M;
  q17.srcPtr.ysize = N;
  q17.dstPtr.ptr = devPitchedg17.ptr;
  q17.dstPtr.pitch = devPitchedg17.pitch;
  q17.dstPtr.xsize = M;
  q17.dstPtr.ysize = N;
  q17.extent.width = M * sizeof(float);
  q17.extent.height = N;
  q17.extent.depth = W;
  q17.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&q17);

  q18.srcPtr.ptr = h_g18;
  q18.srcPtr.pitch = M * sizeof(float);
  q18.srcPtr.xsize = M;
  q18.srcPtr.ysize = N;
  q18.dstPtr.ptr = devPitchedg18.ptr;
  q18.dstPtr.pitch = devPitchedg18.pitch;
  q18.dstPtr.xsize = M;
  q18.dstPtr.ysize = N;
  q18.extent.width = M * sizeof(float);
  q18.extent.height = N;
  q18.extent.depth = W;
  q18.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&q18);

  q0new.srcPtr.ptr = h_g0new;
  q0new.srcPtr.pitch = M * sizeof(float);
  q0new.srcPtr.xsize = M;
  q0new.srcPtr.ysize = N;
  q0new.dstPtr.ptr = devPitchedg0new.ptr;
  q0new.dstPtr.pitch = devPitchedg0new.pitch;
  q0new.dstPtr.xsize = M;
  q0new.dstPtr.ysize = N;
  q0new.extent.width = M * sizeof(float);
  q0new.extent.height = N;
  q0new.extent.depth = W;
  q0new.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&q0new);

  q1new.srcPtr.ptr = h_g1new;
  q1new.srcPtr.pitch = M * sizeof(float);
  q1new.srcPtr.xsize = M;
  q1new.srcPtr.ysize = N;
  q1new.dstPtr.ptr = devPitchedg1new.ptr;
  q1new.dstPtr.pitch = devPitchedg1new.pitch;
  q1new.dstPtr.xsize = M;
  q1new.dstPtr.ysize = N;
  q1new.extent.width = M * sizeof(float);
  q1new.extent.height = N;
  q1new.extent.depth = W;
  q1new.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&q1new);

  q2new.srcPtr.ptr = h_g2new;
  q2new.srcPtr.pitch = M * sizeof(float);
  q2new.srcPtr.xsize = M;
  q2new.srcPtr.ysize = N;
  q2new.dstPtr.ptr = devPitchedg2new.ptr;
  q2new.dstPtr.pitch = devPitchedg2new.pitch;
  q2new.dstPtr.xsize = M;
  q2new.dstPtr.ysize = N;
  q2new.extent.width = M * sizeof(float);
  q2new.extent.height = N;
  q2new.extent.depth = W;
  q2new.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&q2new);

  q3new.srcPtr.ptr = h_g3new;
  q3new.srcPtr.pitch = M * sizeof(float);
  q3new.srcPtr.xsize = M;
  q3new.srcPtr.ysize = N;
  q3new.dstPtr.ptr = devPitchedg3new.ptr;
  q3new.dstPtr.pitch = devPitchedg3new.pitch;
  q3new.dstPtr.xsize = M;
  q3new.dstPtr.ysize = N;
  q3new.extent.width = M * sizeof(float);
  q3new.extent.height = N;
  q3new.extent.depth = W;
  q3new.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&q3new);

  q4new.srcPtr.ptr = h_g4new;
  q4new.srcPtr.pitch = M * sizeof(float);
  q4new.srcPtr.xsize = M;
  q4new.srcPtr.ysize = N;
  q4new.dstPtr.ptr = devPitchedg4new.ptr;
  q4new.dstPtr.pitch = devPitchedg4new.pitch;
  q4new.dstPtr.xsize = M;
  q4new.dstPtr.ysize = N;
  q4new.extent.width = M * sizeof(float);
  q4new.extent.height = N;
  q4new.extent.depth = W;
  q4new.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&q4new);

  q5new.srcPtr.ptr = h_g5new;
  q5new.srcPtr.pitch = M * sizeof(float);
  q5new.srcPtr.xsize = M;
  q5new.srcPtr.ysize = N;
  q5new.dstPtr.ptr = devPitchedg5new.ptr;
  q5new.dstPtr.pitch = devPitchedg5new.pitch;
  q5new.dstPtr.xsize = M;
  q5new.dstPtr.ysize = N;
  q5new.extent.width = M * sizeof(float);
  q5new.extent.height = N;
  q5new.extent.depth = W;
  q5new.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&q5new);

  q6new.srcPtr.ptr = h_g6new;
  q6new.srcPtr.pitch = M * sizeof(float);
  q6new.srcPtr.xsize = M;
  q6new.srcPtr.ysize = N;
  q6new.dstPtr.ptr = devPitchedg6new.ptr;
  q6new.dstPtr.pitch = devPitchedg6new.pitch;
  q6new.dstPtr.xsize = M;
  q6new.dstPtr.ysize = N;
  q6new.extent.width = M * sizeof(float);
  q6new.extent.height = N;
  q6new.extent.depth = W;
  q6new.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&q6new);

  q7new.srcPtr.ptr = h_g7new;
  q7new.srcPtr.pitch = M * sizeof(float);
  q7new.srcPtr.xsize = M;
  q7new.srcPtr.ysize = N;
  q7new.dstPtr.ptr = devPitchedg7new.ptr;
  q7new.dstPtr.pitch = devPitchedg7new.pitch;
  q7new.dstPtr.xsize = M;
  q7new.dstPtr.ysize = N;
  q7new.extent.width = M * sizeof(float);
  q7new.extent.height = N;
  q7new.extent.depth = W;
  q7new.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&q7new);

  q8new.srcPtr.ptr = h_g8new;
  q8new.srcPtr.pitch = M * sizeof(float);
  q8new.srcPtr.xsize = M;
  q8new.srcPtr.ysize = N;
  q8new.dstPtr.ptr = devPitchedg8new.ptr;
  q8new.dstPtr.pitch = devPitchedg8new.pitch;
  q8new.dstPtr.xsize = M;
  q8new.dstPtr.ysize = N;
  q8new.extent.width = M * sizeof(float);
  q8new.extent.height = N;
  q8new.extent.depth = W;
  q8new.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&q8new);

  q9new.srcPtr.ptr = h_g9new;
  q9new.srcPtr.pitch = M * sizeof(float);
  q9new.srcPtr.xsize = M;
  q9new.srcPtr.ysize = N;
  q9new.dstPtr.ptr = devPitchedg9new.ptr;
  q9new.dstPtr.pitch = devPitchedg9new.pitch;
  q9new.dstPtr.xsize = M;
  q9new.dstPtr.ysize = N;
  q9new.extent.width = M * sizeof(float);
  q9new.extent.height = N;
  q9new.extent.depth = W;
  q9new.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&q9new);

  q10new.srcPtr.ptr = h_g10new;
  q10new.srcPtr.pitch = M * sizeof(float);
  q10new.srcPtr.xsize = M;
  q10new.srcPtr.ysize = N;
  q10new.dstPtr.ptr = devPitchedg10new.ptr;
  q10new.dstPtr.pitch = devPitchedg10new.pitch;
  q10new.dstPtr.xsize = M;
  q10new.dstPtr.ysize = N;
  q10new.extent.width = M * sizeof(float);
  q10new.extent.height = N;
  q10new.extent.depth = W;
  q10new.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&q10new);

  q11new.srcPtr.ptr = h_g11new;
  q11new.srcPtr.pitch = M * sizeof(float);
  q11new.srcPtr.xsize = M;
  q11new.srcPtr.ysize = N;
  q11new.dstPtr.ptr = devPitchedg11new.ptr;
  q11new.dstPtr.pitch = devPitchedg11new.pitch;
  q11new.dstPtr.xsize = M;
  q11new.dstPtr.ysize = N;
  q11new.extent.width = M * sizeof(float);
  q11new.extent.height = N;
  q11new.extent.depth = W;
  q11new.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&q11new);

  q12new.srcPtr.ptr = h_g12new;
  q12new.srcPtr.pitch = M * sizeof(float);
  q12new.srcPtr.xsize = M;
  q12new.srcPtr.ysize = N;
  q12new.dstPtr.ptr = devPitchedg12new.ptr;
  q12new.dstPtr.pitch = devPitchedg12new.pitch;
  q12new.dstPtr.xsize = M;
  q12new.dstPtr.ysize = N;
  q12new.extent.width = M * sizeof(float);
  q12new.extent.height = N;
  q12new.extent.depth = W;
  q12new.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&q12new);

  q13new.srcPtr.ptr = h_g13new;
  q13new.srcPtr.pitch = M * sizeof(float);
  q13new.srcPtr.xsize = M;
  q13new.srcPtr.ysize = N;
  q13new.dstPtr.ptr = devPitchedg13new.ptr;
  q13new.dstPtr.pitch = devPitchedg13new.pitch;
  q13new.dstPtr.xsize = M;
  q13new.dstPtr.ysize = N;
  q13new.extent.width = M * sizeof(float);
  q13new.extent.height = N;
  q13new.extent.depth = W;
  q13new.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&q13new);

  q14new.srcPtr.ptr = h_g14new;
  q14new.srcPtr.pitch = M * sizeof(float);
  q14new.srcPtr.xsize = M;
  q14new.srcPtr.ysize = N;
  q14new.dstPtr.ptr = devPitchedg14new.ptr;
  q14new.dstPtr.pitch = devPitchedg14new.pitch;
  q14new.dstPtr.xsize = M;
  q14new.dstPtr.ysize = N;
  q14new.extent.width = M * sizeof(float);
  q14new.extent.height = N;
  q14new.extent.depth = W;
  q14new.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&q14new);

  q15new.srcPtr.ptr = h_g15new;
  q15new.srcPtr.pitch = M * sizeof(float);
  q15new.srcPtr.xsize = M;
  q15new.srcPtr.ysize = N;
  q15new.dstPtr.ptr = devPitchedg15new.ptr;
  q15new.dstPtr.pitch = devPitchedg15new.pitch;
  q15new.dstPtr.xsize = M;
  q15new.dstPtr.ysize = N;
  q15new.extent.width = M * sizeof(float);
  q15new.extent.height = N;
  q15new.extent.depth = W;
  q15new.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&q15new);

  q16new.srcPtr.ptr = h_g16new;
  q16new.srcPtr.pitch = M * sizeof(float);
  q16new.srcPtr.xsize = M;
  q16new.srcPtr.ysize = N;
  q16new.dstPtr.ptr = devPitchedg16new.ptr;
  q16new.dstPtr.pitch = devPitchedg16new.pitch;
  q16new.dstPtr.xsize = M;
  q16new.dstPtr.ysize = N;
  q16new.extent.width = M * sizeof(float);
  q16new.extent.height = N;
  q16new.extent.depth = W;
  q16new.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&q16new);

  q17new.srcPtr.ptr = h_g17new;
  q17new.srcPtr.pitch = M * sizeof(float);
  q17new.srcPtr.xsize = M;
  q17new.srcPtr.ysize = N;
  q17new.dstPtr.ptr = devPitchedg17new.ptr;
  q17new.dstPtr.pitch = devPitchedg17new.pitch;
  q17new.dstPtr.xsize = M;
  q17new.dstPtr.ysize = N;
  q17new.extent.width = M * sizeof(float);
  q17new.extent.height = N;
  q17new.extent.depth = W;
  q17new.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&q17new);

  q18new.srcPtr.ptr = h_g18new;
  q18new.srcPtr.pitch = M * sizeof(float);
  q18new.srcPtr.xsize = M;
  q18new.srcPtr.ysize = N;
  q18new.dstPtr.ptr = devPitchedg18new.ptr;
  q18new.dstPtr.pitch = devPitchedg18new.pitch;
  q18new.dstPtr.xsize = M;
  q18new.dstPtr.ysize = N;
  q18new.extent.width = M * sizeof(float);
  q18new.extent.height = N;
  q18new.extent.depth = W;
  q18new.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&q18new);
}
void LatticeBoltzmann::Collision(void){
  dim3 GridSize_(Mx,My,Mz);
  dim3 BlockSize_(BLOCKSIZE_x,BLOCKSIZE_y,BLOCKSIZE_z);
  d_collition<<<GridSize_,BlockSize_>>>(devPitchedf0,devPitchedf0new,devPitchedg0,devPitchedg0new,
                                        devPitchedf1,devPitchedf1new,devPitchedg1,devPitchedg1new,
                                        devPitchedf2,devPitchedf2new,devPitchedg2,devPitchedg2new,
                                        devPitchedf3,devPitchedf3new,devPitchedg3,devPitchedg3new,
                                        devPitchedf4,devPitchedf4new,devPitchedg4,devPitchedg4new,
                                        devPitchedf5,devPitchedf5new,devPitchedg5,devPitchedg5new,
                                        devPitchedf6,devPitchedf6new,devPitchedg6,devPitchedg6new,
                                        devPitchedf7,devPitchedf7new,devPitchedg7,devPitchedg7new,
                                        devPitchedf8,devPitchedf8new,devPitchedg8,devPitchedg8new,
                                        devPitchedf9,devPitchedf9new,devPitchedg9,devPitchedg9new,
                                        devPitchedf10,devPitchedf10new,devPitchedg10,devPitchedg10new,
                                        devPitchedf11,devPitchedf11new,devPitchedg11,devPitchedg11new,
                                        devPitchedf12,devPitchedf12new,devPitchedg12,devPitchedg12new,
                                        devPitchedf13,devPitchedf13new,devPitchedg13,devPitchedg13new,
                                        devPitchedf14,devPitchedf14new,devPitchedg14,devPitchedg14new,
                                        devPitchedf15,devPitchedf15new,devPitchedg15,devPitchedg15new,
                                        devPitchedf16,devPitchedf16new,devPitchedg16,devPitchedg16new,
                                        devPitchedf17,devPitchedf17new,devPitchedg17,devPitchedg17new,
                                        devPitchedf18,devPitchedf18new,devPitchedg18,devPitchedg18new);
  cudaDeviceSynchronize();
}
void LatticeBoltzmann::Advection(void){
  dim3 GridSize(Mx,My,Mz);
  dim3 BlockSize(BLOCKSIZE_x,BLOCKSIZE_y,BLOCKSIZE_z);
  
  op_indv_advection<<<GridSize,BlockSize>>>(devPitchedf0,devPitchedf0new,0);   op_indv_advection<<<GridSize,BlockSize>>>(devPitchedg0,devPitchedg0new,0);
  op_indv_advection<<<GridSize,BlockSize>>>(devPitchedf1,devPitchedf1new,1);   op_indv_advection<<<GridSize,BlockSize>>>(devPitchedg1,devPitchedg1new,1);
  op_indv_advection<<<GridSize,BlockSize>>>(devPitchedf2,devPitchedf2new,2);   op_indv_advection<<<GridSize,BlockSize>>>(devPitchedg2,devPitchedg2new,2);
  op_indv_advection<<<GridSize,BlockSize>>>(devPitchedf3,devPitchedf3new,3);   op_indv_advection<<<GridSize,BlockSize>>>(devPitchedg3,devPitchedg3new,3);
  op_indv_advection<<<GridSize,BlockSize>>>(devPitchedf4,devPitchedf4new,4);   op_indv_advection<<<GridSize,BlockSize>>>(devPitchedg4,devPitchedg4new,4);
  op_indv_advection<<<GridSize,BlockSize>>>(devPitchedf5,devPitchedf5new,5);   op_indv_advection<<<GridSize,BlockSize>>>(devPitchedg5,devPitchedg5new,5);
  op_indv_advection<<<GridSize,BlockSize>>>(devPitchedf6,devPitchedf6new,6);   op_indv_advection<<<GridSize,BlockSize>>>(devPitchedg6,devPitchedg6new,6);
  op_indv_advection<<<GridSize,BlockSize>>>(devPitchedf7,devPitchedf7new,7);   op_indv_advection<<<GridSize,BlockSize>>>(devPitchedg7,devPitchedg7new,7);
  op_indv_advection<<<GridSize,BlockSize>>>(devPitchedf8,devPitchedf8new,8);   op_indv_advection<<<GridSize,BlockSize>>>(devPitchedg8,devPitchedg8new,8);
  op_indv_advection<<<GridSize,BlockSize>>>(devPitchedf9,devPitchedf9new,9);   op_indv_advection<<<GridSize,BlockSize>>>(devPitchedg9,devPitchedg9new,9);
  op_indv_advection<<<GridSize,BlockSize>>>(devPitchedf10,devPitchedf10new,10);   op_indv_advection<<<GridSize,BlockSize>>>(devPitchedg10,devPitchedg10new,10);
  op_indv_advection<<<GridSize,BlockSize>>>(devPitchedf11,devPitchedf11new,11);   op_indv_advection<<<GridSize,BlockSize>>>(devPitchedg11,devPitchedg11new,11);
  op_indv_advection<<<GridSize,BlockSize>>>(devPitchedf12,devPitchedf12new,12);   op_indv_advection<<<GridSize,BlockSize>>>(devPitchedg12,devPitchedg12new,12);
  op_indv_advection<<<GridSize,BlockSize>>>(devPitchedf13,devPitchedf13new,13);   op_indv_advection<<<GridSize,BlockSize>>>(devPitchedg13,devPitchedg13new,13);
  op_indv_advection<<<GridSize,BlockSize>>>(devPitchedf14,devPitchedf14new,14);   op_indv_advection<<<GridSize,BlockSize>>>(devPitchedg14,devPitchedg14new,14);
  op_indv_advection<<<GridSize,BlockSize>>>(devPitchedf15,devPitchedf15new,15);   op_indv_advection<<<GridSize,BlockSize>>>(devPitchedg15,devPitchedg15new,15);
  op_indv_advection<<<GridSize,BlockSize>>>(devPitchedf16,devPitchedf16new,16);   op_indv_advection<<<GridSize,BlockSize>>>(devPitchedg16,devPitchedg16new,16);
  op_indv_advection<<<GridSize,BlockSize>>>(devPitchedf17,devPitchedf17new,17);   op_indv_advection<<<GridSize,BlockSize>>>(devPitchedg17,devPitchedg17new,17);
  op_indv_advection<<<GridSize,BlockSize>>>(devPitchedf18,devPitchedf18new,18);   op_indv_advection<<<GridSize,BlockSize>>>(devPitchedg18,devPitchedg18new,18);
}

void LatticeBoltzmann::Show(void)
{
  //Devolver al Host
  p0.srcPtr.ptr = devPitchedf0.ptr;
  p0.srcPtr.pitch = devPitchedf0.pitch;
  p0.dstPtr.ptr = h_f0;
  p0.dstPtr.pitch = M * sizeof(float); 
  p0.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&p0);

  p1.srcPtr.ptr = devPitchedf1.ptr;
  p1.srcPtr.pitch = devPitchedf1.pitch;
  p1.dstPtr.ptr = h_f1;
  p1.dstPtr.pitch = M * sizeof(float); 
  p1.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&p1);

  p2.srcPtr.ptr = devPitchedf2.ptr;
  p2.srcPtr.pitch = devPitchedf2.pitch;
  p2.dstPtr.ptr = h_f2;
  p2.dstPtr.pitch = M * sizeof(float); 
  p2.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&p2);

  p3.srcPtr.ptr = devPitchedf3.ptr;
  p3.srcPtr.pitch = devPitchedf3.pitch;
  p3.dstPtr.ptr = h_f3;
  p3.dstPtr.pitch = M * sizeof(float); 
  p3.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&p3);

  p4.srcPtr.ptr = devPitchedf4.ptr;
  p4.srcPtr.pitch = devPitchedf4.pitch;
  p4.dstPtr.ptr = h_f4;
  p4.dstPtr.pitch = M * sizeof(float); 
  p4.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&p4);

  p5.srcPtr.ptr = devPitchedf5.ptr;
  p5.srcPtr.pitch = devPitchedf5.pitch;
  p5.dstPtr.ptr = h_f5;
  p5.dstPtr.pitch = M * sizeof(float); 
  p5.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&p5);

  p6.srcPtr.ptr = devPitchedf6.ptr;
  p6.srcPtr.pitch = devPitchedf6.pitch;
  p6.dstPtr.ptr = h_f6;
  p6.dstPtr.pitch = M * sizeof(float); 
  p6.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&p6);

  p7.srcPtr.ptr = devPitchedf7.ptr;
  p7.srcPtr.pitch = devPitchedf7.pitch;
  p7.dstPtr.ptr = h_f7;
  p7.dstPtr.pitch = M * sizeof(float); 
  p7.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&p7);

  p8.srcPtr.ptr = devPitchedf8.ptr;
  p8.srcPtr.pitch = devPitchedf8.pitch;
  p8.dstPtr.ptr = h_f8;
  p8.dstPtr.pitch = M * sizeof(float); 
  p8.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&p8);

  p9.srcPtr.ptr = devPitchedf9.ptr;
  p9.srcPtr.pitch = devPitchedf9.pitch;
  p9.dstPtr.ptr = h_f9;
  p9.dstPtr.pitch = M * sizeof(float); 
  p9.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&p9);

  p10.srcPtr.ptr = devPitchedf10.ptr;
  p10.srcPtr.pitch = devPitchedf10.pitch;
  p10.dstPtr.ptr = h_f10;
  p10.dstPtr.pitch = M * sizeof(float); 
  p10.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&p10);

  p11.srcPtr.ptr = devPitchedf11.ptr;
  p11.srcPtr.pitch = devPitchedf11.pitch;
  p11.dstPtr.ptr = h_f11;
  p11.dstPtr.pitch = M * sizeof(float); 
  p11.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&p11);

  p12.srcPtr.ptr = devPitchedf12.ptr;
  p12.srcPtr.pitch = devPitchedf12.pitch;
  p12.dstPtr.ptr = h_f12;
  p12.dstPtr.pitch = M * sizeof(float); 
  p12.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&p12);

  p13.srcPtr.ptr = devPitchedf13.ptr;
  p13.srcPtr.pitch = devPitchedf13.pitch;
  p13.dstPtr.ptr = h_f13;
  p13.dstPtr.pitch = M * sizeof(float); 
  p13.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&p13);

  p14.srcPtr.ptr = devPitchedf14.ptr;
  p14.srcPtr.pitch = devPitchedf14.pitch;
  p14.dstPtr.ptr = h_f14;
  p14.dstPtr.pitch = M * sizeof(float); 
  p14.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&p14);


  p15.srcPtr.ptr = devPitchedf15.ptr;
  p15.srcPtr.pitch = devPitchedf15.pitch;
  p15.dstPtr.ptr = h_f15;
  p15.dstPtr.pitch = M * sizeof(float); 
  p15.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&p15);

  p16.srcPtr.ptr = devPitchedf16.ptr;
  p16.srcPtr.pitch = devPitchedf16.pitch;
  p16.dstPtr.ptr = h_f16;
  p16.dstPtr.pitch = M * sizeof(float); 
  p16.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&p16);

  p17.srcPtr.ptr = devPitchedf17.ptr;
  p17.srcPtr.pitch = devPitchedf17.pitch;
  p17.dstPtr.ptr = h_f17;
  p17.dstPtr.pitch = M * sizeof(float); 
  p17.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&p17);


  p18.srcPtr.ptr = devPitchedf18.ptr;
  p18.srcPtr.pitch = devPitchedf18.pitch;
  p18.dstPtr.ptr = h_f18;
  p18.dstPtr.pitch = M * sizeof(float); 
  p18.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&p18);

  q0.srcPtr.ptr = devPitchedg0.ptr;
  q0.srcPtr.pitch = devPitchedg0.pitch;
  q0.dstPtr.ptr = h_g0;
  q0.dstPtr.pitch = M * sizeof(float); 
  q0.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&q0);

  q1.srcPtr.ptr = devPitchedg1.ptr;
  q1.srcPtr.pitch = devPitchedg1.pitch;
  q1.dstPtr.ptr = h_g1;
  q1.dstPtr.pitch = M * sizeof(float); 
  q1.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&q1);

  q2.srcPtr.ptr = devPitchedg2.ptr;
  q2.srcPtr.pitch = devPitchedg2.pitch;
  q2.dstPtr.ptr = h_g2;
  q2.dstPtr.pitch = M * sizeof(float); 
  q2.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&q2);

  q3.srcPtr.ptr = devPitchedg3.ptr;
  q3.srcPtr.pitch = devPitchedg3.pitch;
  q3.dstPtr.ptr = h_g3;
  q3.dstPtr.pitch = M * sizeof(float); 
  q3.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&q3);

  q4.srcPtr.ptr = devPitchedg4.ptr;
  q4.srcPtr.pitch = devPitchedg4.pitch;
  q4.dstPtr.ptr = h_g4;
  q4.dstPtr.pitch = M * sizeof(float); 
  q4.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&q4);

  q5.srcPtr.ptr = devPitchedg5.ptr;
  q5.srcPtr.pitch = devPitchedg5.pitch;
  q5.dstPtr.ptr = h_g5;
  q5.dstPtr.pitch = M * sizeof(float); 
  q5.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&q5);

  q6.srcPtr.ptr = devPitchedg6.ptr;
  q6.srcPtr.pitch = devPitchedg6.pitch;
  q6.dstPtr.ptr = h_g6;
  q6.dstPtr.pitch = M * sizeof(float); 
  q6.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&q6);

  q7.srcPtr.ptr = devPitchedg7.ptr;
  q7.srcPtr.pitch = devPitchedg7.pitch;
  q7.dstPtr.ptr = h_g7;
  q7.dstPtr.pitch = M * sizeof(float); 
  q7.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&q7);

  q8.srcPtr.ptr = devPitchedg8.ptr;
  q8.srcPtr.pitch = devPitchedg8.pitch;
  q8.dstPtr.ptr = h_g8;
  q8.dstPtr.pitch = M * sizeof(float); 
  q8.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&q8);

  q9.srcPtr.ptr = devPitchedg9.ptr;
  q9.srcPtr.pitch = devPitchedg9.pitch;
  q9.dstPtr.ptr = h_g9;
  q9.dstPtr.pitch = M * sizeof(float); 
  q9.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&q9);

  q10.srcPtr.ptr = devPitchedg10.ptr;
  q10.srcPtr.pitch = devPitchedg10.pitch;
  q10.dstPtr.ptr = h_g10;
  q10.dstPtr.pitch = M * sizeof(float); 
  q10.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&q10);

  q11.srcPtr.ptr = devPitchedg11.ptr;
  q11.srcPtr.pitch = devPitchedg11.pitch;
  q11.dstPtr.ptr = h_g11;
  q11.dstPtr.pitch = M * sizeof(float); 
  q11.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&q11);

  q12.srcPtr.ptr = devPitchedg12.ptr;
  q12.srcPtr.pitch = devPitchedg12.pitch;
  q12.dstPtr.ptr = h_g12;
  q12.dstPtr.pitch = M * sizeof(float); 
  q12.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&q12);

  q13.srcPtr.ptr = devPitchedg13.ptr;
  q13.srcPtr.pitch = devPitchedg13.pitch;
  q13.dstPtr.ptr = h_g13;
  q13.dstPtr.pitch = M * sizeof(float); 
  q13.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&q13);

  q14.srcPtr.ptr = devPitchedg14.ptr;
  q14.srcPtr.pitch = devPitchedg14.pitch;
  q14.dstPtr.ptr = h_g14;
  q14.dstPtr.pitch = M * sizeof(float); 
  q14.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&q14);


  q15.srcPtr.ptr = devPitchedg15.ptr;
  q15.srcPtr.pitch = devPitchedg15.pitch;
  q15.dstPtr.ptr = h_g15;
  q15.dstPtr.pitch = M * sizeof(float); 
  q15.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&q15);

  q16.srcPtr.ptr = devPitchedg16.ptr;
  q16.srcPtr.pitch = devPitchedg16.pitch;
  q16.dstPtr.ptr = h_g16;
  q16.dstPtr.pitch = M * sizeof(float); 
  q16.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&q16);

  q17.srcPtr.ptr = devPitchedg17.ptr;
  q17.srcPtr.pitch = devPitchedg17.pitch;
  q17.dstPtr.ptr = h_g17;
  q17.dstPtr.pitch = M * sizeof(float); 
  q17.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&q17);


  q18.srcPtr.ptr = devPitchedg18.ptr;
  q18.srcPtr.pitch = devPitchedg18.pitch;
  q18.dstPtr.ptr = h_g18;
  q18.dstPtr.pitch = M * sizeof(float); 
  q18.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&q18);

  //Devolver al Host
  p0new.srcPtr.ptr = devPitchedf0new.ptr;
  p0new.srcPtr.pitch = devPitchedf0new.pitch;
  p0new.dstPtr.ptr = h_f0new;
  p0new.dstPtr.pitch = M * sizeof(float); 
  p0new.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&p0new);

  p1new.srcPtr.ptr = devPitchedf1new.ptr;
  p1new.srcPtr.pitch = devPitchedf1new.pitch;
  p1new.dstPtr.ptr = h_f1new;
  p1new.dstPtr.pitch = M * sizeof(float); 
  p1new.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&p1new);

  p2new.srcPtr.ptr = devPitchedf2new.ptr;
  p2new.srcPtr.pitch = devPitchedf2new.pitch;
  p2new.dstPtr.ptr = h_f2new;
  p2new.dstPtr.pitch = M * sizeof(float); 
  p2new.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&p2new);

  p3new.srcPtr.ptr = devPitchedf3new.ptr;
  p3new.srcPtr.pitch = devPitchedf3new.pitch;
  p3new.dstPtr.ptr = h_f3new;
  p3new.dstPtr.pitch = M * sizeof(float); 
  p3new.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&p3new);

  p4new.srcPtr.ptr = devPitchedf4new.ptr;
  p4new.srcPtr.pitch = devPitchedf4new.pitch;
  p4new.dstPtr.ptr = h_f4new;
  p4new.dstPtr.pitch = M * sizeof(float); 
  p4new.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&p4new);

  p5new.srcPtr.ptr = devPitchedf5new.ptr;
  p5new.srcPtr.pitch = devPitchedf5new.pitch;
  p5new.dstPtr.ptr = h_f5new;
  p5new.dstPtr.pitch = M * sizeof(float); 
  p5new.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&p5new);

  p6new.srcPtr.ptr = devPitchedf6new.ptr;
  p6new.srcPtr.pitch = devPitchedf6new.pitch;
  p6new.dstPtr.ptr = h_f6new;
  p6new.dstPtr.pitch = M * sizeof(float); 
  p6new.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&p6new);

  p7new.srcPtr.ptr = devPitchedf7new.ptr;
  p7new.srcPtr.pitch = devPitchedf7new.pitch;
  p7new.dstPtr.ptr = h_f7new;
  p7new.dstPtr.pitch = M * sizeof(float); 
  p7new.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&p7new);

  p8new.srcPtr.ptr = devPitchedf8new.ptr;
  p8new.srcPtr.pitch = devPitchedf8new.pitch;
  p8new.dstPtr.ptr = h_f8new;
  p8new.dstPtr.pitch = M * sizeof(float); 
  p8new.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&p8new);

  p9new.srcPtr.ptr = devPitchedf9new.ptr;
  p9new.srcPtr.pitch = devPitchedf9new.pitch;
  p9new.dstPtr.ptr = h_f9new;
  p9new.dstPtr.pitch = M * sizeof(float); 
  p9new.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&p9new);

  p10new.srcPtr.ptr = devPitchedf10new.ptr;
  p10new.srcPtr.pitch = devPitchedf10new.pitch;
  p10new.dstPtr.ptr = h_f10new;
  p10new.dstPtr.pitch = M * sizeof(float); 
  p10new.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&p10new);

  p11new.srcPtr.ptr = devPitchedf11new.ptr;
  p11new.srcPtr.pitch = devPitchedf11new.pitch;
  p11new.dstPtr.ptr = h_f11new;
  p11new.dstPtr.pitch = M * sizeof(float); 
  p11new.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&p11new);

  p12new.srcPtr.ptr = devPitchedf12new.ptr;
  p12new.srcPtr.pitch = devPitchedf12new.pitch;
  p12new.dstPtr.ptr = h_f12new;
  p12new.dstPtr.pitch = M * sizeof(float); 
  p12new.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&p12new);

  p13new.srcPtr.ptr = devPitchedf13new.ptr;
  p13new.srcPtr.pitch = devPitchedf13new.pitch;
  p13new.dstPtr.ptr = h_f13new;
  p13new.dstPtr.pitch = M * sizeof(float); 
  p13new.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&p13new);

  p14new.srcPtr.ptr = devPitchedf14new.ptr;
  p14new.srcPtr.pitch = devPitchedf14new.pitch;
  p14new.dstPtr.ptr = h_f14new;
  p14new.dstPtr.pitch = M * sizeof(float); 
  p14new.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&p14new);


  p15new.srcPtr.ptr = devPitchedf15new.ptr;
  p15new.srcPtr.pitch = devPitchedf15new.pitch;
  p15new.dstPtr.ptr = h_f15new;
  p15new.dstPtr.pitch = M * sizeof(float); 
  p15new.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&p15new);

  p16new.srcPtr.ptr = devPitchedf16new.ptr;
  p16new.srcPtr.pitch = devPitchedf16new.pitch;
  p16new.dstPtr.ptr = h_f16new;
  p16new.dstPtr.pitch = M * sizeof(float); 
  p16new.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&p16new);

  p17new.srcPtr.ptr = devPitchedf17new.ptr;
  p17new.srcPtr.pitch = devPitchedf17new.pitch;
  p17new.dstPtr.ptr = h_f17new;
  p17new.dstPtr.pitch = M * sizeof(float); 
  p17new.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&p17new);


  p18new.srcPtr.ptr = devPitchedf18new.ptr;
  p18new.srcPtr.pitch = devPitchedf18new.pitch;
  p18new.dstPtr.ptr = h_f18new;
  p18new.dstPtr.pitch = M * sizeof(float); 
  p18new.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&p18new);

  q0new.srcPtr.ptr = devPitchedg0new.ptr;
  q0new.srcPtr.pitch = devPitchedg0new.pitch;
  q0new.dstPtr.ptr = h_g0new;
  q0new.dstPtr.pitch = M * sizeof(float); 
  q0new.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&q0new);

  q1new.srcPtr.ptr = devPitchedg1new.ptr;
  q1new.srcPtr.pitch = devPitchedg1new.pitch;
  q1new.dstPtr.ptr = h_g1new;
  q1new.dstPtr.pitch = M * sizeof(float); 
  q1new.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&q1new);

  q2new.srcPtr.ptr = devPitchedg2new.ptr;
  q2new.srcPtr.pitch = devPitchedg2new.pitch;
  q2new.dstPtr.ptr = h_g2new;
  q2new.dstPtr.pitch = M * sizeof(float); 
  q2new.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&q2new);

  q3new.srcPtr.ptr = devPitchedg3new.ptr;
  q3new.srcPtr.pitch = devPitchedg3new.pitch;
  q3new.dstPtr.ptr = h_g3new;
  q3new.dstPtr.pitch = M * sizeof(float); 
  q3new.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&q3new);

  q4new.srcPtr.ptr = devPitchedg4new.ptr;
  q4new.srcPtr.pitch = devPitchedg4new.pitch;
  q4new.dstPtr.ptr = h_g4new;
  q4new.dstPtr.pitch = M * sizeof(float); 
  q4new.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&q4new);

  q5new.srcPtr.ptr = devPitchedg5new.ptr;
  q5new.srcPtr.pitch = devPitchedg5new.pitch;
  q5new.dstPtr.ptr = h_g5new;
  q5new.dstPtr.pitch = M * sizeof(float); 
  q5new.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&q5new);

  q6new.srcPtr.ptr = devPitchedg6new.ptr;
  q6new.srcPtr.pitch = devPitchedg6new.pitch;
  q6new.dstPtr.ptr = h_g6new;
  q6new.dstPtr.pitch = M * sizeof(float); 
  q6new.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&q6new);

  q7new.srcPtr.ptr = devPitchedg7new.ptr;
  q7new.srcPtr.pitch = devPitchedg7new.pitch;
  q7new.dstPtr.ptr = h_g7new;
  q7new.dstPtr.pitch = M * sizeof(float); 
  q7new.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&q7new);

  q8new.srcPtr.ptr = devPitchedg8new.ptr;
  q8new.srcPtr.pitch = devPitchedg8new.pitch;
  q8new.dstPtr.ptr = h_g8new;
  q8new.dstPtr.pitch = M * sizeof(float); 
  q8new.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&q8new);

  q9new.srcPtr.ptr = devPitchedg9new.ptr;
  q9new.srcPtr.pitch = devPitchedg9new.pitch;
  q9new.dstPtr.ptr = h_g9new;
  q9new.dstPtr.pitch = M * sizeof(float); 
  q9new.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&q9new);

  q10new.srcPtr.ptr = devPitchedg10new.ptr;
  q10new.srcPtr.pitch = devPitchedg10new.pitch;
  q10new.dstPtr.ptr = h_g10new;
  q10new.dstPtr.pitch = M * sizeof(float); 
  q10new.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&q10new);

  q11new.srcPtr.ptr = devPitchedg11new.ptr;
  q11new.srcPtr.pitch = devPitchedg11new.pitch;
  q11new.dstPtr.ptr = h_g11new;
  q11new.dstPtr.pitch = M * sizeof(float); 
  q11new.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&q11new);

  q12new.srcPtr.ptr = devPitchedg12new.ptr;
  q12new.srcPtr.pitch = devPitchedg12new.pitch;
  q12new.dstPtr.ptr = h_g12new;
  q12new.dstPtr.pitch = M * sizeof(float); 
  q12new.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&q12new);

  q13new.srcPtr.ptr = devPitchedg13new.ptr;
  q13new.srcPtr.pitch = devPitchedg13new.pitch;
  q13new.dstPtr.ptr = h_g13new;
  q13new.dstPtr.pitch = M * sizeof(float); 
  q13new.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&q13new);

  q14new.srcPtr.ptr = devPitchedg14new.ptr;
  q14new.srcPtr.pitch = devPitchedg14new.pitch;
  q14new.dstPtr.ptr = h_g14new;
  q14new.dstPtr.pitch = M * sizeof(float); 
  q14new.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&q14new);


  q15new.srcPtr.ptr = devPitchedg15new.ptr;
  q15new.srcPtr.pitch = devPitchedg15new.pitch;
  q15new.dstPtr.ptr = h_g15new;
  q15new.dstPtr.pitch = M * sizeof(float); 
  q15new.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&q15new);

  q16new.srcPtr.ptr = devPitchedg16new.ptr;
  q16new.srcPtr.pitch = devPitchedg16new.pitch;
  q16new.dstPtr.ptr = h_g16new;
  q16new.dstPtr.pitch = M * sizeof(float); 
  q16new.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&q16new);

  q17new.srcPtr.ptr = devPitchedg17new.ptr;
  q17new.srcPtr.pitch = devPitchedg17new.pitch;
  q17new.dstPtr.ptr = h_g17new;
  q17new.dstPtr.pitch = M * sizeof(float); 
  q17new.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&q17new);


  q18new.srcPtr.ptr = devPitchedg18new.ptr;
  q18new.srcPtr.pitch = devPitchedg18new.pitch;
  q18new.dstPtr.ptr = h_g18new;
  q18new.dstPtr.pitch = M * sizeof(float); 
  q18new.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&q18new);
/*
    for (int w=0; w<W; w++) 
      for (int j=0; j<N; j++) 
        for (int i=0; i<M; i++){
         cout << h_g10[w][j][i] << " ";
         if(i == M -1){
           cout << endl;
         }
        }
    cout << endl;
*/
}


float LatticeBoltzmann::h_Ux(int ix,int iy,int iz){
  float sum1=0, sum2=0;
  
  sum2 = h_g0[iz][iy][ix]*h_Vx[0]+h_g1[iz][iy][ix]*h_Vx[1]+h_g2[iz][iy][ix]*h_Vx[2]+h_g3[iz][iy][ix]*h_Vx[3]+h_g4[iz][iy][ix]*h_Vx[4]+h_g5[iz][iy][ix]*h_Vx[5]+h_g6[iz][iy][ix]*h_Vx[6]+h_g7[iz][iy][ix]*h_Vx[7]+h_g8[iz][iy][ix]*h_Vx[8]+h_g9[iz][iy][ix]*h_Vx[9]+h_g10[iz][iy][ix]*h_Vx[10]+h_g11[iz][iy][ix]*h_Vx[11]+h_g12[iz][iy][ix]*h_Vx[12]+h_g13[iz][iy][ix]*h_Vx[13]+h_g14[iz][iy][ix]*h_Vx[14]+h_g15[iz][iy][ix]*h_Vx[15]+h_g16[iz][iy][ix]*h_Vx[16]+h_g17[iz][iy][ix]*h_Vx[17]+h_g18[iz][iy][ix]*h_Vx[18];
  
  sum1 = h_g0[iz][iy][ix]+h_g1[iz][iy][ix]+h_g2[iz][iy][ix]+h_g3[iz][iy][ix]+h_g4[iz][iy][ix]+h_g5[iz][iy][ix]+h_g6[iz][iy][ix]+h_g7[iz][iy][ix]+h_g8[iz][iy][ix]+h_g9[iz][iy][ix]+h_g10[iz][iy][ix]+h_g11[iz][iy][ix]+h_g12[iz][iy][ix]+h_g13[iz][iy][ix]+h_g14[iz][iy][ix]+h_g15[iz][iy][ix]+h_g16[iz][iy][ix]+h_g17[iz][iy][ix]+h_g18[iz][iy][ix];
  
  return 3.*sum2/(3.*sum1 + 3.*h_P(ix,iy,iz));
}
float LatticeBoltzmann::h_Uy(int ix,int iy,int iz){
  float sum1=0, sum2=0;
  
  sum2 = h_g0[iz][iy][ix]*h_Vy[0]+h_g1[iz][iy][ix]*h_Vy[1]+h_g2[iz][iy][ix]*h_Vy[2]+h_g3[iz][iy][ix]*h_Vy[3]+h_g4[iz][iy][ix]*h_Vy[4]+h_g5[iz][iy][ix]*h_Vy[5]+h_g6[iz][iy][ix]*h_Vy[6]+h_g7[iz][iy][ix]*h_Vy[7]+h_g8[iz][iy][ix]*h_Vy[8]+h_g9[iz][iy][ix]*h_Vy[9]+h_g10[iz][iy][ix]*h_Vy[10]+h_g11[iz][iy][ix]*h_Vy[11]+h_g12[iz][iy][ix]*h_Vy[12]+h_g13[iz][iy][ix]*h_Vy[13]+h_g14[iz][iy][ix]*h_Vy[14]+h_g15[iz][iy][ix]*h_Vy[15]+h_g16[iz][iy][ix]*h_Vy[16]+h_g17[iz][iy][ix]*h_Vy[17]+h_g18[iz][iy][ix]*h_Vy[18];
  
  sum1 = h_g0[iz][iy][ix]+h_g1[iz][iy][ix]+h_g2[iz][iy][ix]+h_g3[iz][iy][ix]+h_g4[iz][iy][ix]+h_g5[iz][iy][ix]+h_g6[iz][iy][ix]+h_g7[iz][iy][ix]+h_g8[iz][iy][ix]+h_g9[iz][iy][ix]+h_g10[iz][iy][ix]+h_g11[iz][iy][ix]+h_g12[iz][iy][ix]+h_g13[iz][iy][ix]+h_g14[iz][iy][ix]+h_g15[iz][iy][ix]+h_g16[iz][iy][ix]+h_g17[iz][iy][ix]+h_g18[iz][iy][ix];
  
  return 3.*sum2/(3.*sum1 + 3.*h_P(ix,iy,iz));
}
float LatticeBoltzmann::h_Uz(int ix,int iy,int iz){
  float sum1=0, sum2=0;
  
  sum2 = h_g0[iz][iy][ix]*h_Vz[0]+h_g1[iz][iy][ix]*h_Vz[1]+h_g2[iz][iy][ix]*h_Vz[2]+h_g3[iz][iy][ix]*h_Vz[3]+h_g4[iz][iy][ix]*h_Vz[4]+h_g5[iz][iy][ix]*h_Vz[5]+h_g6[iz][iy][ix]*h_Vz[6]+h_g7[iz][iy][ix]*h_Vz[7]+h_g8[iz][iy][ix]*h_Vz[8]+h_g9[iz][iy][ix]*h_Vz[9]+h_g10[iz][iy][ix]*h_Vz[10]+h_g11[iz][iy][ix]*h_Vz[11]+h_g12[iz][iy][ix]*h_Vz[12]+h_g13[iz][iy][ix]*h_Vz[13]+h_g14[iz][iy][ix]*h_Vz[14]+h_g15[iz][iy][ix]*h_Vz[15]+h_g16[iz][iy][ix]*h_Vz[16]+h_g17[iz][iy][ix]*h_Vz[17]+h_g18[iz][iy][ix]*h_Vz[18];
  
  sum1 = h_g0[iz][iy][ix]+h_g1[iz][iy][ix]+h_g2[iz][iy][ix]+h_g3[iz][iy][ix]+h_g4[iz][iy][ix]+h_g5[iz][iy][ix]+h_g6[iz][iy][ix]+h_g7[iz][iy][ix]+h_g8[iz][iy][ix]+h_g9[iz][iy][ix]+h_g10[iz][iy][ix]+h_g11[iz][iy][ix]+h_g12[iz][iy][ix]+h_g13[iz][iy][ix]+h_g14[iz][iy][ix]+h_g15[iz][iy][ix]+h_g16[iz][iy][ix]+h_g17[iz][iy][ix]+h_g18[iz][iy][ix];
  
  return 3.*sum2/(3.*sum1 + 3.*h_P(ix,iy,iz));
}
float LatticeBoltzmann::h_gamma(float Ux0,float Uy0,float Uz0){
  float U2;
  U2 = Ux0*Ux0 + Uy0*Uy0 + Uz0*Uz0;
  return 1./sqrt(1.-(U2/(C*C)));
}
float LatticeBoltzmann::h_n(int ix,int iy,int iz,float Ux0,float Uy0,float Uz0){
  float sum = 0;
  sum = h_f0[iz][iy][ix]+h_f1[iz][iy][ix]+h_f2[iz][iy][ix]+h_f3[iz][iy][ix]+h_f4[iz][iy][ix]+h_f5[iz][iy][ix]+h_f6[iz][iy][ix]+h_f7[iz][iy][ix]+h_f8[iz][iy][ix]+h_f9[iz][iy][ix]+h_f10[iz][iy][ix]+h_f11[iz][iy][ix]+h_f12[iz][iy][ix]+h_f13[iz][iy][ix]+h_f14[iz][iy][ix]+h_f15[iz][iy][ix]+h_f16[iz][iy][ix]+h_f17[iz][iy][ix]+h_f18[iz][iy][ix];
  return sum/h_gamma(Ux0,Uy0,Uz0);
}
float LatticeBoltzmann::h_P(int ix,int iy,int iz){
  int i,j; float sum1=0, sum2=0;
  float g_aux[19] = {h_g0[iz][iy][ix],h_g1[iz][iy][ix],h_g2[iz][iy][ix],h_g3[iz][iy][ix],h_g4[iz][iy][ix],h_g5[iz][iy][ix],h_g6[iz][iy][ix],h_g7[iz][iy][ix],h_g8[iz][iy][ix],h_g9[iz][iy][ix],h_g10[iz][iy][ix],h_g11[iz][iy][ix],h_g12[iz][iy][ix],h_g13[iz][iy][ix],h_g14[iz][iy][ix],h_g15[iz][iy][ix],h_g16[iz][iy][ix],h_g17[iz][iy][ix],h_g18[iz][iy][ix]};

  for(i=0;i<Q;i++){
    sum1 += g_aux[i];
    for(j=0;j<Q;j++){
      sum2 += (g_aux[i]*g_aux[j]*(h_Vx[i]*h_Vx[j]+h_Vy[i]*h_Vy[j]+h_Vz[i]*h_Vz[j]));
    }
  }
  return -(1./3.)*sum1 + (1./3.)*sqrt(-3.*sum2 + 4.*sum1*sum1);
}
float LatticeBoltzmann::h_rho(int ix,int iy,int iz){
  return 3.*h_P(ix,iy,iz);
}
float LatticeBoltzmann::h_feq(int i,float n0,float Ux0,float Uy0,float Uz0){
  float y,U2,UdotV;

  y = h_gamma(Ux0,Uy0,Uz0);
  UdotV = Ux0*h_Vx[i]+Uy0*h_Vy[i]+Uz0*h_Vz[i];
  U2 = Ux0*Ux0 + Uy0*Uy0 + Uz0*Uz0;

  return h_w[i]*n0*y*(1.+3.*UdotV/(cl*cl) + (9./2.)*(UdotV*UdotV)/(cl*cl*cl*cl) - (3./2.)*(U2/(cl*cl)));
}
float LatticeBoltzmann::h_geq(int i,float rho0,float P0,float Ux0,float Uy0,float Uz0){
  float y2,UdotV,U2;

  y2 = h_gamma(Ux0,Uy0,Uz0)*h_gamma(Ux0,Uy0,Uz0);
  UdotV = Ux0*h_Vx[i]+Uy0*h_Vy[i]+Uz0*h_Vz[i];
  U2 = Ux0*Ux0 + Uy0*Uy0 + Uz0*Uz0;
  
  if(i == 0){
    return 3.*P0*h_w[0]*y2*(4. - (2.+ cl*cl)/(y2*cl*cl) - 2.*(U2/(cl*cl)));
  }else{
    return 3.*h_w[i]*P0*y2*( 1./(y2*cl*cl) + 4.*UdotV/(cl*cl) + 6.*(UdotV*UdotV)/(cl*cl*cl*cl) - 2.*(U2/(cl*cl)) );
  }
}

void LatticeBoltzmann::Print(const char * NombreArchivo){
  float Ux0,Uy0,Uz0;
  //Imprimir en un archivo
  ofstream MiArchivo(NombreArchivo);
  ofstream X_Y("X_Y_cut.dat");
  ofstream X_Z("X_Z_cut.dat");
  Show();
  for(int ix=0;ix<M;ix++){
    for(int iy=0;iy<N;iy++)
      for(int iz=0;iz<W;iz++){
        Ux0=h_Ux(ix,iy,iz);
        Uy0=h_Uy(ix,iy,iz);
        Uz0=h_Uz(ix,iy,iz);
        MiArchivo<<ix<<" "<< iy << " " << iz << " " << h_n(ix,iy,iz,Ux0,Uy0,Uz0)<<" "<<h_P(ix,iy,iz)/2.495e-7<<endl;
        if(iz == int(W*0.3)){
          X_Y<<ix<<" "<< iy << " " << h_P(ix,iy,iz)/2.495e-7 << endl;
        }else if(iy == int(N*0.3)){
          X_Z<<ix<<" "<< iz << " " << h_P(ix,iy,iz)/2.495e-7 << endl;
        }
	}
    MiArchivo<<endl;
    X_Y<<endl;
    X_Z<<endl;
  }
  MiArchivo.close();
  X_Y.close();
  X_Z.close();
}


//----------------------------------------------------------

int main()
{ LatticeBoltzmann Relativistic_Ang;

  float Ux0 = 0.0;
  float Uy0 = 0.0;
  float Uz0 = 0.0;

  float T = 0.0314;
  float dg = 16;

  float P0 = 2.495e-7;
  float P1 = 1.023e-7;

  float n0 = P0/T;
  float n1 = P1/T;

  float rho0 = 3*n0*T;
  float rho1 = 3*n1*T;

  int t,tmax = 5000;

  Relativistic_Ang.Start(Ux0,Uy0,Uz0,rho0,rho1,n0,n1,P0,P1);
  
  for(t=0;t<tmax;t++){
    Relativistic_Ang.Collision();
    Relativistic_Ang.Advection();
  }
  Relativistic_Ang.Print("data.dat");   
  return 0;
}
