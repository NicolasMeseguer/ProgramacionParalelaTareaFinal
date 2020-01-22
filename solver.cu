#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>
#include "wtime.h"
#include "definitions.h"
#include "energy_struct.h"
#include "cuda_runtime.h"
#include "solver.h"
#include "cuda.h"

using namespace std;

/**
* Kernel del calculo de la solvation. Se debe anadir los parametros 
*/
__global__ void escalculation (int atoms_r, int atoms_l, int nlig, float *rec_x_d, float *rec_y_d, float *rec_z_d, float *lig_x_d, float *lig_y_d, float *lig_z_d, float *ql_d,float *qr_d, float *energy_d, int nconformations){
  
  int ind = blockIdx.x*blockDim.x+threadIdx.x;
  int i = ind * nlig;
  double dist, total_elec = 0, miatomo[3], elecTerm;	
  if(ind < nconformations){	
    for(int j = 0; j < atoms_l; j++){
      miatomo[0] = *(lig_x_d + i + j);
      miatomo[1] = *(lig_y_d + i + j);
      miatomo[2] = *(lig_z_d + i + j);
      for(int k = 0; k < atoms_r; k++){
        elecTerm = 0;
        dist = calculaDistancia(rec_x_d[k], rec_y_d[k], rec_z_d[k], miatomo[0], miatomo[1],miatomo[2]);
        elecTerm = (ql_d[j]*qr_d[k]) / dist;
        total_elec += elecTerm;
      }
    }
    energy_d[i/nlig] = total_elec;
    total_elec = 0;
  }
}

/**
* Funcion para manejar el lanzamiento de CUDA 
*/
void forces_GPU_AU (int atoms_r, int atoms_l, int nlig, float *rec_x, float *rec_y, float *rec_z, float *lig_x, float *lig_y, float *lig_z, float *ql ,float *qr, float *energy, int nconformations){
	
	cudaError_t cudaStatus; //variable para recoger estados de cuda


	//seleccionamos device
	cudaSetDevice(0); //0 - Tesla K40 vs 1 - Tesla K230

	//creamos memoria para los vectores para GPU _d (device)
	float *rec_x_d, *rec_y_d, *rec_z_d, *qr_d, *lig_x_d, *lig_y_d, *lig_z_d, *ql_d, *energy_d;

	//reservamos memoria para GPU
  int memsize;
  
  //Diferente TAM de memoria... Explicacion en la entrevista si es necesario.
  //memsize = sizeof(float)*nlig;
  memsize = sizeof(float)*nlig*nconformations;
  cudaMalloc(&lig_x_d, memsize);
  cudaMalloc(&lig_y_d, memsize);
  cudaMalloc(&lig_z_d, memsize);
  cudaMalloc(&ql_d, memsize);
  
  memsize = sizeof(float)*atoms_r;
  cudaMalloc(&rec_x_d, memsize);
  cudaMalloc(&rec_y_d, memsize);
  cudaMalloc(&rec_z_d, memsize);
  cudaMalloc(&qr_d, memsize);
	
  memsize = sizeof(float)*nconformations;
  cudaMalloc(&energy_d, memsize);
  
//pasamos datos de host to device
  cudaMemcpy(energy_d, energy, memsize, cudaMemcpyHostToDevice);
  
  memsize = sizeof(float)*atoms_r;
  cudaMemcpy(rec_x_d, rec_x, memsize, cudaMemcpyHostToDevice);
  cudaMemcpy(rec_y_d, rec_y, memsize, cudaMemcpyHostToDevice);
  cudaMemcpy(rec_z_d, rec_z, memsize, cudaMemcpyHostToDevice);
  cudaMemcpy(qr_d, qr, memsize, cudaMemcpyHostToDevice);
  
  memsize = sizeof(float)*nlig*nconformations;
  cudaMemcpy(lig_x_d, lig_x, memsize, cudaMemcpyHostToDevice);
  cudaMemcpy(lig_y_d, lig_y, memsize, cudaMemcpyHostToDevice);
  cudaMemcpy(lig_z_d, lig_z, memsize, cudaMemcpyHostToDevice);
  cudaMemcpy(ql_d, ql, memsize, cudaMemcpyHostToDevice);

	//Definir numero de hilos y bloques
	double TAMBLOCK = 256;
  int block = ceilf(((double)nconformations)/TAMBLOCK);
  int thread = (TAMBLOCK);
  printf("Bloques: %d\n", block);
	printf("Hilos por bloque: %d\n", thread);

	//llamamos a kernel
	escalculation <<< block,thread>>> (atoms_r, atoms_l, nlig, rec_x_d, rec_y_d, rec_z_d, lig_x_d, lig_y_d, lig_z_d, ql_d, qr_d, energy_d, nconformations);
	
	//control de errores kernel
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
  if(cudaStatus != cudaSuccess) fprintf(stderr, "Error en el kernel %d\n", cudaStatus); 

	
  //Traemos info al host 
  /*Esta info no hace falta
  cudaMemcpy(lig_x, lig_x_d, memsize, cudaMemcpyDeviceToHost);
  cudaMemcpy(lig_y, lig_y_d, memsize, cudaMemcpyDeviceToHost);
  cudaMemcpy(lig_z, lig_z_d, memsize, cudaMemcpyDeviceToHost);
  cudaMemcpy(ql, ql_d, memsize, cudaMemcpyDeviceToHost);
	
  memsize = sizeof(float)*atoms_r;
  cudaMemcpy(rec_x, rec_x_d, memsize, cudaMemcpyDeviceToHost);
  cudaMemcpy(rec_y, rec_y_d, memsize, cudaMemcpyDeviceToHost);
  cudaMemcpy(rec_z, rec_z_d, memsize, cudaMemcpyDeviceToHost);
  cudamemcpy(qr, qr_d, memsize, cudaMemcpyDeviceToHost);
  */
  memsize = sizeof(float)*nconformations;
  cudaMemcpy(energy, energy_d, memsize, cudaMemcpyDeviceToHost);  

  // para comprobar que la ultima conformacion tiene el mismo resultado que la primera
	printf("Termino electrostatico de conformacion %d es: %f\n", nconformations-1, energy[nconformations-1]); 

	//resultado varia repecto a SECUENCIAL y CUDA en 0.000002 por falta de precision con float
	//posible solucion utilizar double, probablemente bajara el rendimiento -> mas tiempo para calculo
	printf("Termino electrostatico %f\n", energy[0]);

	//Liberamos memoria reservada para GPU
  cudaFree(rec_x_d);cudaFree(rec_y_d);cudaFree(rec_z_d);cudaFree(lig_x_d);cudaFree(lig_y_d);cudaFree(lig_z_d);cudaFree(qr_d);cudaFree(ql_d);cudaFree(energy_d);
}

/**
* Distancia euclidea compartida por funcion CUDA y CPU secuencial
*/
__device__ __host__ extern float calculaDistancia (float rx, float ry, float rz, float lx, float ly, float lz) {

  float difx = rx - lx;
  float dify = ry - ly;
  float difz = rz - lz;
  float mod2x=difx*difx;
  float mod2y=dify*dify;
  float mod2z=difz*difz;
  difx=mod2x+mod2y+mod2z;
  return sqrtf(difx);
}




/**
 * Funcion que implementa el termino electrostático en CPU
 */
void forces_CPU_AU (int atoms_r, int atoms_l, int nlig, float *rec_x, float *rec_y, float *rec_z, float *lig_x, float *lig_y, float *lig_z, float *ql ,float *qr, float *energy, int nconformations){

	double dist, total_elec = 0, miatomo[3], elecTerm;
  int totalAtomLig = nconformations * nlig;

	for (int k=0; k < totalAtomLig; k+=nlig){
	  for(int i=0;i<atoms_l;i++){					
			miatomo[0] = *(lig_x + k + i);
			miatomo[1] = *(lig_y + k + i);
			miatomo[2] = *(lig_z + k + i);

			for(int j=0;j<atoms_r;j++){				
				elecTerm = 0;
        dist=calculaDistancia (rec_x[j], rec_y[j], rec_z[j], miatomo[0], miatomo[1], miatomo[2]);
//				printf ("La distancia es %lf\n", dist);
        elecTerm = (ql[i]* qr[j]) / dist;
				total_elec += elecTerm;
//        printf ("La carga es %lf\n", total_elec);
			}
		}
		
		energy[k/nlig] = total_elec;
		total_elec = 0;
  }
	printf("Termino electrostatico %f\n", energy[0]);
}


extern void solver_AU(int mode, int atoms_r, int atoms_l,  int nlig, float *rec_x, float *rec_y, float *rec_z, float *lig_x, float *lig_y, float *lig_z, float *ql, float *qr, float *energy_desolv, int nconformaciones) {

	double elapsed_i, elapsed_o;
	
	switch (mode) {
		case 0://Sequential execution
			printf("\* CALCULO ELECTROSTATICO EN CPU *\n");
			printf("**************************************\n");			
			printf("Conformations: %d\t Mode: %d, CPU\n",nconformaciones,mode);			
			elapsed_i = wtime();
			forces_CPU_AU (atoms_r,atoms_l,nlig,rec_x,rec_y,rec_z,lig_x,lig_y,lig_z,ql,qr,energy_desolv,nconformaciones);
			elapsed_o = wtime() - elapsed_i;
			printf ("CPU Processing time: %f (seg)\n", elapsed_o);
			break;
		case 1: //OpenMP execution
			printf("\* CALCULO ELECTROSTATICO EN OPENMP *\n");
			printf("**************************************\n");			
			printf("**************************************\n");			
			printf("Conformations: %d\t Mode: %d, CMP\n",nconformaciones,mode);			
			elapsed_i = wtime();
			forces_OMP_AU (atoms_r,atoms_l,nlig,rec_x,rec_y,rec_z,lig_x,lig_y,lig_z,ql,qr,energy_desolv,nconformaciones);
			elapsed_o = wtime() - elapsed_i;
			printf ("OpenMP Processing time: %f (seg)\n", elapsed_o);
			break;
		case 2: //CUDA exeuction
			printf("\* CALCULO ELECTROSTATICO EN CUDA *\n");
      printf("**************************************\n");
      printf("Conformaciones: %d\t Mode: %d, GPU\n",nconformaciones,mode);
			elapsed_i = wtime();
			forces_GPU_AU (atoms_r,atoms_l,nlig,rec_x,rec_y,rec_z,lig_x,lig_y,lig_z,ql,qr,energy_desolv,nconformaciones);
			elapsed_o = wtime() - elapsed_i;
			printf ("GPU Processing time: %f (seg)\n", elapsed_o);			
			break; 	
	  	default:
 	    	printf("Wrong mode type: %d.  Use -h for help.\n", mode);
			exit (-1);	
	} 		
}
