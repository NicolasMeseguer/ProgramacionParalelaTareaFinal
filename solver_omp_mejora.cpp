#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>
#include "wtime.h"
#include "definitions.h"
#include "energy_struct.h"
#include "solver.h"

/**
* Funcion que implementa la solvatacion en openmp
*/
extern void forces_OMP_AU (int atoms_r, int atoms_l, int nlig, float *rec_x, float *rec_y, float *rec_z, float *lig_x, float *lig_y, float *lig_z, float *ql ,float *qr, float *energy, int nconformations){

  float dist, total_elec = 0, miatomo[3], elecTerm;
  int totalAtomLig = nconformations * nlig;
  
  #pragma omp parallel for private (elecTerm, dist, total_elec) num_threads(2)
  for (int k=0; k < totalAtomLig; k+=nlig) {
    total_elec = 0;
    for(int j=0;j<atoms_r;j++){
	    for(int i=0;i<atoms_l;i++){
		    miatomo[0] = *(lig_x + k + i);
        miatomo[1] = *(lig_y + k + i);
        miatomo[2] = *(lig_z + k + i);
        elecTerm = 0;
        dist=calculaDistancia (rec_x[j], rec_y[j], rec_z[j], miatomo[0], miatomo[1], miatomo[2]);
        elecTerm = (ql[i]* qr[j]) / dist;
        total_elec += elecTerm;
      }
    }
    energy[k/nlig] = total_elec;
  }
  
  printf("Termino electrostatico %f\n", energy[0]);
}



