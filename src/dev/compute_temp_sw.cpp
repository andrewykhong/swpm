/* ----------------------------------------------------------------------
   SPARTA - Stochastic PArallel Rarefied-gas Time-accurate Analyzer
   http://sparta.sandia.gov
   Steve Plimpton, sjplimp@gmail.com, Michael Gallis, magalli@sandia.gov
   Sandia National Laboratories

   Copyright (2014) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level SPARTA directory.
------------------------------------------------------------------------- */

#include "mpi.h"
#include "string.h"
#include "compute_temp_sw.h"
#include "update.h"
#include "particle.h"
#include "domain.h"
#include "error.h"

using namespace SPARTA_NS;

/* ---------------------------------------------------------------------- */

ComputeTempSW::ComputeTempSW(SPARTA *sparta, int narg, char **arg) :
  Compute(sparta, narg, arg)
{
  if (narg != 2) error->all(FLERR,"Illegal compute temp command");

  scalar_flag = 1;
}

/* ---------------------------------------------------------------------- */

double ComputeTempSW::compute_scalar()
{
  invoked_scalar = update->ntimestep;

  Particle::Species *species = particle->species;
  Particle::OnePart *particles = particle->particles;
  int nlocal = particle->nlocal;

  double *v;
  double t = 0.0;
  double g;
  double sumg = 0.0;
  double sumg_global;

  for (int i = 0; i < nlocal; i++) {
    v = particles[i].v;
    g = particles[i].sw;
    t += (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]) *
      species[particles[i].ispecies].mass*g;
    sumg += g;
  }

  MPI_Allreduce(&t,&scalar,1,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(&sumg,&sumg_global,1,MPI_DOUBLE,MPI_SUM,world);

  bigint n = particle->nlocal;
  MPI_Allreduce(&n,&particle->nglobal,1,MPI_SPARTA_BIGINT,MPI_SUM,world);
  if (particle->nglobal == 0) return 0.0;

  // normalize with 3 instead of dim since even 2d has 3 velocity components

  double factor = update->mvv2e / (3.0 * sumg_global * update->boltz);
  scalar *= factor;
  return scalar;
}
