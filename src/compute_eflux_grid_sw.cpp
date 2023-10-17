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

#include "string.h"
#include "compute_eflux_grid_sw.h"
#include "particle.h"
#include "mixture.h"
#include "grid.h"
#include "update.h"
#include "modify.h"
#include "memory.h"
#include "error.h"

using namespace SPARTA_NS;

// user keywords

enum{HEATX,HEATY,HEATZ};

// internal accumulators

//enum{MASSSUM,mVx,mVy,mVz,mVxVx,mVyVy,mVzVz,mVxVy,mVyVz,mVxVz,
//     mVxVxVx,mVyVyVy,mVzVzVz,
//     mVxVyVy,mVxVzVz,mVyVxVx,mVyVzVz,mVzVxVx,mVzVyVy,
//     Np,sG,LASTSIZE};

enum{gSum,gVx,gVy,gVz,
      gVxVx, gVyVy, gVzVz,
      gVxVy, gVxVz, gVyVz,
      gVxVV, gVyVV, gVzVV,
      LASTSIZE};

// max # of quantities to accumulate for any user value

#define MAXACCUMULATE 12

/* ---------------------------------------------------------------------- */

ComputeEFluxGridSW::ComputeEFluxGridSW(SPARTA *sparta, int narg, char **arg) :
  Compute(sparta, narg, arg)
{
  if (narg < 5) error->all(FLERR,"Illegal compute eflux/grid/sw command");

  int igroup = grid->find_group(arg[2]);
  if (igroup < 0)
    error->all(FLERR,"Compute eflux/grid/sw group ID does not exist");
  groupbit = grid->bitmask[igroup];

  imix = particle->find_mixture(arg[3]);
  if (imix < 0) error->all(FLERR,"Compute eflux/grid/sw mixture ID "
                           "does not exist");
  ngroup = particle->mixture[imix]->ngroup;

  nvalue = narg - 4;
  value = new int[nvalue];

  npergroup = 0;
  unique = new int[LASTSIZE];
  nmap = new int[nvalue];
  memory->create(map,ngroup*nvalue,MAXACCUMULATE,"eflux/grid/sw:map");
  for (int i = 0; i < nvalue; i++) nmap[i] = 0;

  int ivalue = 0;
  int iarg = 4;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"heatx") == 0) {
      value[ivalue] = HEATX;
      set_map(ivalue,gSum);
      set_map(ivalue,gVx);
      set_map(ivalue,gVy);
      set_map(ivalue,gVz);
      set_map(ivalue,gVxVx);
      set_map(ivalue,gVyVy);
      set_map(ivalue,gVzVz);
      set_map(ivalue,gVxVy);
      set_map(ivalue,gVxVz);
      set_map(ivalue,gVyVz);
      set_map(ivalue,gVxVV);
    } else if (strcmp(arg[iarg],"heaty") == 0) {
      value[ivalue] = HEATY;
      set_map(ivalue,gSum);
      set_map(ivalue,gVy);
      set_map(ivalue,gVz);
      set_map(ivalue,gVx);
      set_map(ivalue,gVyVy);
      set_map(ivalue,gVzVz);
      set_map(ivalue,gVxVx);
      set_map(ivalue,gVyVz);
      set_map(ivalue,gVxVy);
      set_map(ivalue,gVxVz);
      set_map(ivalue,gVyVV);
    } else if (strcmp(arg[iarg],"heatz") == 0) {
      value[ivalue] = HEATZ;
      set_map(ivalue,gSum);
      set_map(ivalue,gVz);
      set_map(ivalue,gVx);
      set_map(ivalue,gVy);
      set_map(ivalue,gVzVz);
      set_map(ivalue,gVxVx);
      set_map(ivalue,gVyVy);
      set_map(ivalue,gVxVz);
      set_map(ivalue,gVyVz);
      set_map(ivalue,gVxVy);
      set_map(ivalue,gVzVV);
    } else error->all(FLERR,"Illegal compute eflux/grid/sw command");

    ivalue++;
    iarg++;
  }

  // ntotal = total # of columns in tally array
  // reset_map() adjusts indices in initial map() using final npergroup
  // also adds columns to tally array for CELLCOUNT/CELLMASS

  ntotal = ngroup*npergroup;
  reset_map();

  per_grid_flag = 1;
  size_per_grid_cols = ngroup*nvalue;
  post_process_grid_flag = 1;

  nglocal = 0;
  vector_grid = NULL;
  tally = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeEFluxGridSW::~ComputeEFluxGridSW()
{
  if (copymode) return;

  delete [] value;
  delete [] unique;

  delete [] nmap;
  memory->destroy(map);

  memory->destroy(vector_grid);
  memory->destroy(tally);
}

/* ---------------------------------------------------------------------- */

void ComputeEFluxGridSW::init()
{
  if (ngroup != particle->mixture[imix]->ngroup)
    error->all(FLERR,"Number of groups in compute eflux/grid/sw mixture "
               "has changed");

  reallocate();
}

/* ---------------------------------------------------------------------- */

void ComputeEFluxGridSW::compute_per_grid()
{
  invoked_per_grid = update->ntimestep;

  Grid::ChildInfo *cinfo = grid->cinfo;
  Particle::Species *species = particle->species;
  Particle::OnePart *particles = particle->particles;
  int *s2g = particle->mixture[imix]->species2group;
  int nlocal = particle->nlocal;

  int i,j,k,m,ispecies,igroup,icell;
  double mass, vsq;
  double *v,*vec;
  double g;

  // zero all accumulators - could do this with memset()
  for (i = 0; i < nglocal; i++)
    for (j = 0; j < ntotal; j++)
      tally[i][j] = 0.0;

  // calculate bulk
  for (i = 0; i < nlocal; i++) {
    ispecies = particles[i].ispecies;
    igroup = s2g[ispecies];
    if (igroup < 0) continue;
    icell = particles[i].icell;
    if (!(cinfo[icell].mask & groupbit)) continue;

    mass = species[ispecies].mass;
    v = particles[i].v;
    g = particles[i].sw;
    vsq = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];

    vec = tally[icell];

    // loop has all possible values particle needs to accumulate
    // subset defined by user values are indexed by accumulate vector
    // NOTE: at some point may need prefactors v,v^2,v^3 converted to p,eng

    k = igroup*npergroup;
    for (m = 0; m < npergroup; m++) {
      switch (unique[m]) {
      case gSum:
        vec[k++] += mass*g;
        break;
      case gVx:
        vec[k++] += mass*v[0]*g;
        break;
      case gVy:
        vec[k++] += mass*v[1]*g;
        break;
      case gVz:
        vec[k++] += mass*v[2]*g;
        break;
      case gVxVx:
        vec[k++] += mass*v[0]*v[0]*g;
        break;
      case gVyVy:
        vec[k++] += mass*v[1]*v[1]*g;
        break;
      case gVzVz:
        vec[k++] += mass*v[2]*v[2]*g;
        break;
      case gVxVy:
        vec[k++] += mass*v[0]*v[1]*g;
        break;
      case gVxVz:
        vec[k++] += mass*v[0]*v[2]*g;
        break;
      case gVyVz:
        vec[k++] += mass*v[1]*v[2]*g;
        break;
      case gVxVV:
        vec[k++] += mass*v[0]*vsq*g*0.5;
        break;
      case gVyVV:
        vec[k++] += mass*v[1]*vsq*g*0.5;
        break;
      case gVzVV:
        vec[k++] += mass*v[2]*vsq*g*0.5;
        break;
      }
    }
  }
}

/* ----------------------------------------------------------------------
   query info about internal tally array for this compute
   index = which column of output (0 for vec, 1 to N for array)
   return # of tally quantities for this index
   also return array = ptr to tally array
   also return cols = ptr to list of columns in tally for this index
------------------------------------------------------------------------- */

int ComputeEFluxGridSW::query_tally_grid(int index, double **&array, int *&cols)
{
  index--;
  int ivalue = index % nvalue;
  array = tally;
  cols = map[index];
  return nmap[ivalue];
}

/* ----------------------------------------------------------------------
   tally accumulated info to compute final normalized values
   index = which column of output (0 for vec, 1 to N for array)
   for etally = NULL:
     use internal tallied info for single timestep, set nsample = 1
     compute values for all grid cells
       store results in vector_grid with nstride = 1 (single col of array_grid)
   for etally = ptr to caller array:
     use external tallied info for many timesteps
     nsample = additional normalization factor used by some values
     emap = list of etally columns to use, # of columns determined by index
     store results in caller's vec, spaced by nstride
   if norm = 0.0, set result to 0.0 directly so do not divide by 0.0
------------------------------------------------------------------------- */

void ComputeEFluxGridSW::post_process_grid(int index, int nsample,
                                         double **etally, int *emap,
                                         double *vec, int nstride)
{
  index--;
  int ivalue = index % nvalue;

  int lo = 0;
  int hi = nglocal;
  int k = 0;

  if (!etally) {
    nsample = 1;
    etally = tally;
    emap = map[index];
    vec = vector_grid;
    nstride = 1;
  }

  double *t;
  Grid::ChildInfo *cinfo = grid->cinfo;

  int mass = emap[0];
  int mv   = emap[1];
  int mv1  = emap[2];
  int mv2  = emap[3];

  int mvv   = emap[4];
  int mv1v1 = emap[5];
  int mv2v2 = emap[6];

  int mvv1  = emap[7];
  int mvv2  = emap[8];
  int mv1v2 = emap[9];

  int mvvv  = emap[10];

  double wt;
  double V, V1, V2;
  double pvv, pv1v1, pv2v2;
  double pvv1, pvv2, pv1v2;
  double Vsq, T;

  for (int icell = lo; icell < hi; icell++) {
    t = etally[icell];
    V = t[mv]/t[mass];
    V1 = t[mv1]/t[mass];
    V2 = t[mv2]/t[mass];

    pvv = t[mvv] - t[mass]*V*V;
    pv1v1 = t[mv1v1] - t[mass]*V1*V1;
    pv2v2 = t[mv2v2] - t[mass]*V2*V2;
    pvv1 = t[mvv1] - t[mass]*V*V1;
    pvv2 = t[mvv2] - t[mass]*V*V2;
    pv1v2 = t[mv1v2] - t[mass]*V1*V2;

    Vsq = V*V + V1*V1 + V2*V2;
    T = (pvv + pv1v1 + pv2v2) / (3.0 * t[mass]);
 
    wt = cinfo[icell].weight / cinfo[icell].volume;

    vec[k] = wt/nsample * (t[mvvv] - (pvv*V + pvv1*V1 + pvv2*V2) -
      0.5*t[mass]*V*Vsq - 1.5*t[mass]*T*V);
    k += nstride;
  }
}

/* ----------------------------------------------------------------------
   add a tally quantity to all groups for ivalue
   also add it to unique list if first time this name is used
   name = name of tally quantity from enum{} at top of file
   nmap[i] = # of tally quantities for user value I
   map[i][k] = index of Kth tally quantity for output value I
   npergroup = length of unique list
------------------------------------------------------------------------- */

void ComputeEFluxGridSW::set_map(int ivalue, int name)
{
  // index = loc of name in current unique list if there, else npergroup

  int index = 0;
  for (index = 0; index < npergroup; index++)
    if (unique[index] == name) break;

  // if name is not already in unique, add it and increment npergroup

  if (index == npergroup) {
    index = npergroup;
    unique[npergroup++] = name;
  }

  // add index to map and nmap for all groups
  // will add group offset in reset_map()

  for (int igroup = 0; igroup < ngroup; igroup++)
    map[igroup*nvalue+ivalue][nmap[ivalue]] = index;
  nmap[ivalue]++;
}

/* ----------------------------------------------------------------------
   reset map indices to reflect final npergroup = unique quantities/group
------------------------------------------------------------------------- */

void ComputeEFluxGridSW::reset_map()
{
  for (int i = 0; i < ngroup*nvalue; i++) {
    int igroup = i / nvalue;
    int ivalue = i % nvalue;
    for (int k = 0; k < nmap[ivalue]; k++)
      map[i][k] += igroup*npergroup;
  }
}

/* ----------------------------------------------------------------------
   reallocate data storage if nglocal has changed
   called by init() and whenever grid changes
------------------------------------------------------------------------- */

void ComputeEFluxGridSW::reallocate()
{
  if (grid->nlocal == nglocal) return;

  memory->destroy(vector_grid);
  memory->destroy(tally);
  nglocal = grid->nlocal;
  memory->create(vector_grid,nglocal,"grid:vector_grid");
  memory->create(tally,nglocal,ntotal,"grid:tally");
}

/* ----------------------------------------------------------------------
   memory usage of local grid-based data
------------------------------------------------------------------------- */

bigint ComputeEFluxGridSW::memory_usage()
{
  bigint bytes;
  bytes = nglocal * sizeof(double);
  bytes = ntotal*nglocal * sizeof(double);
  return bytes;
}
