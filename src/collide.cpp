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

#include "math.h"
#include "string.h"
#include "collide.h"
#include "particle.h"
#include "mixture.h"
#include "update.h"
#include "grid.h"
#include "comm.h"
#include "react.h"
#include "modify.h"
#include "fix.h"
#include "fix_ambipolar.h"
#include "random_mars.h"
#include "random_knuth.h"
#include "memory.h"
#include "error.h"
#include "math_const.h"
#include "math_eigen.h"
#include "math_eigen_impl.h"
#include "math_extra.h"

#include <iostream>
#include <fstream>
using namespace std;

using namespace SPARTA_NS;

enum{NONE,DISCRETE,SMOOTH};       // several files  (NOTE: change order)
enum{PKEEP,PINSERT,PDONE,PDISCARD,PENTRY,PEXIT,PSURF};   // several files

#define DELTAGRID 1000            // must be bigger than split cells per cell
#define DELTADELETE 1024
#define DELTAELECTRON 128
#define DELTAGRP 8
#define SMALLNUM 1.0e-8

#define BIG 1.0e20

/* ---------------------------------------------------------------------- */

Collide::Collide(SPARTA *sparta, int, char **arg) : Pointers(sparta)
{
  int n = strlen(arg[0]) + 1;
  style = new char[n];
  strcpy(style,arg[0]);

  n = strlen(arg[1]) + 1;
  mixID = new char[n];
  strcpy(mixID,arg[1]);

  random = new RanKnuth(update->ranmaster->uniform());
  double seed = update->ranmaster->uniform();
  random->reset(seed,comm->me,100);

  ngroups = 0;

  npmax = 0;
  plist = NULL;
  p2g = NULL;

  nglocal = nglocalmax = 0;

  ngroup = NULL;
  maxgroup = NULL;
  glist = NULL;
  gpair = NULL;

  maxdelete = 0;
  dellist = NULL;

  vre_first = 1;
  vre_start = 1;
  vre_every = 0;
  remainflag = 1;
  vremax = NULL;
  vremax_initial = NULL;
  remain = NULL;
  rotstyle = SMOOTH;
  vibstyle = NONE;
  nearcp = 0;
  nearlimit = 10;

  recomb_ijflag = NULL;

  ambiflag = 0;
  maxelectron = 0;
  elist = NULL;

  // used if near-neighbor model is invoked

  max_nn = 1;
  memory->create(nn_last_partner,max_nn,"collide:nn_last_partner");
  memory->create(nn_last_partner_igroup,max_nn,"collide:nn_last_partner");
  memory->create(nn_last_partner_jgroup,max_nn,"collide:nn_last_partner");

  // initialize counters in case stats outputs them

  ncollide_one = nattempt_one = nreact_one = 0;
  ncollide_running = nattempt_running = nreact_running = 0;

  copymode = kokkos_flag = 0;

  // SWPM - General
  swpmflag = 0;
  Nmin = 0;
  Nmax = -1;
  //gmax_initial = 0;
  //gmax = NULL;
  Ggamma = 1;
  reduction_type = -1;

  // SWPM - Clustering
  currentCluster = NULL;
  npmx = 0;
  rFlag = 0;
  npThresh = 0;
  reduceMinFlag = -1;
  gthFlag = 0; // gsum or npmx

  // SWPM - Reduction
  memory->create(ipij,3,3,"collide:ipij");
  memory->create(ie,3,"collide:ie");
  memory->create(iq,3,"collide:iq");
  memory->create(iV,3,"collide:iV");
  memory->create(evec,3,3,"collide:evec");
  memory->create(eval,3,"collide:eval");
  memory->create(Rij,3,3,"collide:Rij");
  memory->create(xmr,3,"collide:xmr");
  
  // hard coded
  //subNx = 1;
  //subNy = 4;
  //subNz = 1;
  //subN = subNx*subNy*subNz;
  //subCell = NULL;
  //subCellN = NULL;
}

/* ---------------------------------------------------------------------- */

Collide::~Collide()
{
  if (copymode) return;

  delete [] style;
  delete [] mixID;
  delete random;

  memory->destroy(plist);
  memory->destroy(p2g);

  if (ngroups > 1) {
    delete [] ngroup;
    delete [] maxgroup;
    for (int i = 0; i < ngroups; i++) memory->destroy(glist[i]);
    delete [] glist;
    memory->destroy(gpair);
  }

  memory->destroy(dellist);
  memory->sfree(elist);
  memory->destroy(vremax);
  memory->destroy(vremax_initial);
  memory->destroy(remain);
  memory->destroy(nn_last_partner);
  memory->destroy(nn_last_partner_igroup);
  memory->destroy(nn_last_partner_jgroup);

  memory->destroy(recomb_ijflag);

  // SWPM
  memory->destroy(currentCluster);

  memory->destroy(ipij);
  memory->destroy(ie);
  memory->destroy(iq);
  memory->destroy(iV);
  memory->destroy(evec);
  memory->destroy(eval);
  memory->destroy(Rij);
  memory->destroy(xmr);
  //memory->destroy(subCell);
  //memory->destroy(subCellN);
}

/* ---------------------------------------------------------------------- */

void Collide::init()
{
  // error check
  if (ambiflag && nearcp)
    error->all(FLERR,"Ambipolar collision model does not yet support "
               "near-neighbor collisions");

  // require mixture to contain all species

  int imix = particle->find_mixture(mixID);
  if (imix < 0) error->all(FLERR,"Collision mixture does not exist");
  mixture = particle->mixture[imix];

  if (mixture->nspecies != particle->nspecies)
    error->all(FLERR,"Collision mixture does not contain all species");

  if (sparta->kokkos && !kokkos_flag)
    error->all(FLERR,"Must use Kokkos-supported collision style if "
               "Kokkos is enabled");

  // if rotstyle or vibstyle = DISCRETE,
  // check that extra rotation/vibration info is defined
  // for species that require it

  if (rotstyle == DISCRETE) {
    Particle::Species *species = particle->species;
    int nspecies = particle->nspecies;

    int flag = 0;
    for (int isp = 0; isp < nspecies; isp++) {
      if (species[isp].rotdof == 0) continue;
      if (species[isp].rotdof == 2 && species[isp].nrottemp != 1) flag++;
      if (species[isp].rotdof == 3 && species[isp].nrottemp != 3) flag++;
    }
    if (flag) {
      char str[128];
      sprintf(str,"%d species do not define correct rotational "
              "temps for discrete model",flag);
      error->all(FLERR,str);
    }
  }

  if (vibstyle == DISCRETE) {
    index_vibmode = particle->find_custom((char *) "vibmode");

    Particle::Species *species = particle->species;
    int nspecies = particle->nspecies;

    int flag = 0;
    for (int isp = 0; isp < nspecies; isp++) {
      if (species[isp].vibdof <= 2) continue;
      if (index_vibmode < 0)
        error->all(FLERR,
                   "Fix vibmode must be used with discrete vibrational modes");
      if (species[isp].nvibmode != species[isp].vibdof / 2) flag++;
    }
    if (flag) {
      char str[128];
      sprintf(str,"%d species do not define correct vibrational "
              "modes for discrete model",flag);
      error->all(FLERR,str);
    }
  }

  // reallocate one-cell data structs for one or many groups

  oldgroups = ngroups;
  ngroups = mixture->ngroup;

  if (ngroups != oldgroups) {
    if (oldgroups == 1) {
      memory->destroy(plist);
      npmax = 0;
      plist = NULL;
    }
    if (oldgroups > 1) {
      delete [] ngroup;
      delete [] maxgroup;
      for (int i = 0; i < oldgroups; i++) memory->destroy(glist[i]);
      delete [] glist;
      memory->destroy(gpair);
      ngroup = NULL;
      maxgroup = NULL;
      glist = NULL;
      gpair = NULL;
    }

    if (ngroups == 1) {
      npmax = DELTAPART;
      memory->create(plist,npmax,"collide:plist");
    }
    if (ngroups > 1) {
      ngroup = new int[ngroups];
      maxgroup = new int[ngroups];
      glist = new int*[ngroups];
      for (int i = 0; i < ngroups; i++) {
        maxgroup[i] = DELTAPART;
        memory->create(glist[i],DELTAPART,"collide:glist");
      }
      memory->create(gpair,ngroups*ngroups,3,"collide:gpair");
    }
  }

  // allocate vremax,remain if group count changed
  // will always be allocated on first run since oldgroups = 0
  // set vremax_intitial via values calculated by collide style

  if (ngroups != oldgroups) {
    memory->destroy(vremax);
    memory->destroy(vremax_initial);
    memory->destroy(remain);
    nglocal = grid->nlocal;
    nglocalmax = nglocal;
    memory->create(vremax,nglocalmax,ngroups,ngroups,"collide:vremax");
    memory->create(vremax_initial,ngroups,ngroups,"collide:vremax_initial");
    if (remainflag)
      memory->create(remain,nglocalmax,ngroups,ngroups,"collide:remain");

    for (int igroup = 0; igroup < ngroups; igroup++)
      for (int jgroup = 0; jgroup < ngroups; jgroup++)
        vremax_initial[igroup][jgroup] = vremax_init(igroup,jgroup);
  }

  // if recombination reactions exist, set flags per species pair

  recombflag = 0;
  if (react) {
    recombflag = react->recombflag;
    recomb_boost_inverse = react->recomb_boost_inverse;
  }

  if (recombflag) {
    int nspecies = particle->nspecies;
    memory->destroy(recomb_ijflag);
    memory->create(recomb_ijflag,nspecies,nspecies,"collide:recomb_ijflag");
    for (int i = 0; i < nspecies; i++)
      for (int j = 0; j < nspecies; j++)
        recomb_ijflag[i][j] = react->recomb_exist(i,j);
  }

  // find ambipolar fix
  // set ambipolar vector/array indices
  // if reactions defined, check that they are valid ambipolar reactions

  if (ambiflag) {
    index_ionambi = particle->find_custom((char *) "ionambi");
    index_velambi = particle->find_custom((char *) "velambi");
    if (index_ionambi < 0 || index_velambi < 0)
      error->all(FLERR,"Collision ambipolar without fix ambipolar");
    if (react) react->ambi_check();

    int ifix;
    for (ifix = 0; ifix < modify->nfix; ifix++)
      if (strcmp(modify->fix[ifix]->style,"ambipolar") == 0) break;
    FixAmbipolar *afix = (FixAmbipolar *) modify->fix[ifix];
    ambispecies = afix->especies;
  }

  // if ambipolar and multiple groups in mixture, ambispecies must be its own group

  if (ambiflag && mixture->ngroup > 1) {
    int *species2group = mixture->species2group;
    int egroup = species2group[ambispecies];
    if (mixture->groupsize[egroup] != 1)
      error->all(FLERR,"Multigroup ambipolar collisions require "
                 "electrons be their own group");
  }

  // vre_next = next timestep to zero vremax & remain, based on vre_every

  if (vre_every) vre_next = (update->ntimestep/vre_every)*vre_every + vre_every;
  else vre_next = update->laststep + 1;

  // if requested reset vremax & remain
  // must be after per-species vremax_initial is setup

  if (vre_first || vre_start) {
    reset_vremax();
    vre_first = 0;
  }

  // initialize running stats before each run
  ncollide_running = nattempt_running = nreact_running = 0;

  if(swpmflag) {
    maxCluster = npmx;
    memory->create(currentCluster,maxCluster,"collide:currentCluster");
    //memory->create(subCell,subN,npmax,"collide:subCell");
    //memory->create(subCellN,subN,"collide:subCellN");
  }

}

/* ---------------------------------------------------------------------- */

void Collide::modify_params(int narg, char **arg)
{
  if (narg == 0) error->all(FLERR,"Illegal collide_modify command");

  int iarg = 0;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"vremax") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal collide_modify command");
      vre_every = atoi(arg[iarg+1]);
      if (vre_every < 0) error->all(FLERR,"Illegal collide_modify command");
      if (strcmp(arg[iarg+2],"yes") == 0) vre_start = 1;
      else if (strcmp(arg[iarg+2],"no") == 0) vre_start = 0;
      else error->all(FLERR,"Illegal collide_modify command");
      iarg += 3;
    } else if (strcmp(arg[iarg],"remain") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal collide_modify command");
      if (strcmp(arg[iarg+1],"yes") == 0) remainflag = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) remainflag = 0;
      else error->all(FLERR,"Illegal collide_modify command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"rotate") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal collide_modify command");
      if (strcmp(arg[iarg+1],"no") == 0) rotstyle = NONE;
      // not yet supported
      //else if (strcmp(arg[iarg+1],"discrete") == 0) rotstyle = DISCRETE;
      else if (strcmp(arg[iarg+1],"smooth") == 0) rotstyle = SMOOTH;
      else error->all(FLERR,"Illegal collide_modify command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"vibrate") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal collide_modify command");
      if (strcmp(arg[iarg+1],"no") == 0) vibstyle = NONE;
      else if (strcmp(arg[iarg+1],"discrete") == 0) vibstyle = DISCRETE;
      else if (strcmp(arg[iarg+1],"smooth") == 0) vibstyle = SMOOTH;
      else error->all(FLERR,"Illegal collide_modify command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"ambipolar") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal collide_modify command");
      if (strcmp(arg[iarg+1],"no") == 0) ambiflag = 0;
      else if (strcmp(arg[iarg+1],"yes") == 0) ambiflag = 1;
      else error->all(FLERR,"Illegal collide_modify command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"nearcp") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal collide_modify command");
      if (strcmp(arg[iarg+1],"yes") == 0) nearcp = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) nearcp = 0;
      else error->all(FLERR,"Illegal collide_modify command");
      nearlimit = atoi(arg[iarg+2]);
      if (nearcp && nearlimit <= 0)
        error->all(FLERR,"Illegal collide_modify command");
      iarg += 3;
    } else if (strcmp(arg[iarg],"reduceMin") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal collide_modify command");
      if (strcmp(arg[iarg+1],"yes") == 0) reduceMinFlag = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) reduceMinFlag = -1;
      else error->all(FLERR,"Illegal collide_modify command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"swpm") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal collide_modify command");
      if (strcmp(arg[iarg+1],"no") == 0) swpmflag = 0;
      else if (strcmp(arg[iarg+1],"yes") == 0) swpmflag = 1;
      else error->all(FLERR,"Illegal collide_modify command");
      // Thresholds for when splitting or merging is used
      Nmin = atoi(arg[iarg+2]);
      Ggamma = atof(arg[iarg+3]);
      if(Ggamma < 0) Ggamma = 1;
      iarg += 4;
    } else if (strcmp(arg[iarg],"ENRGY") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal collide_modify command");
      reduction_type = 0;
      npThresh = 2;
      Nmax = atoi(arg[iarg+1]); // max before merge
      npmx = atof(arg[iarg+2]); // max per group
      if(npmx <= npThresh) error->all(FLERR,"Too few particles per group");
      iarg += 3;
    } else if (strcmp(arg[iarg],"RW") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal collide_modify command");
      reduction_type = 1;
      npThresh = 2;
      Nmax = atoi(arg[iarg+1]); // max before merge
      npmx = atof(arg[iarg+2]); // max per group
      if(npmx <= npThresh) error->all(FLERR,"Too few particles per group");
      iarg += 3;
    } else if (strcmp(arg[iarg],"LGZ") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal collide_modify command");
      reduction_type = 2;
      npThresh = 6;
      Nmax = atoi(arg[iarg+1]); // max before merge
      npmx = atof(arg[iarg+2]); // max per group
      if(npmx <= npThresh) error->all(FLERR,"Too few particles per group");
      iarg += 3;
    } else if (strcmp(arg[iarg],"GONO") == 0) {
      reduction_type = 3;
      if (iarg+7 > narg) error->all(FLERR,"Illegal collide_modify command");
      reduction_type = 3;
      Nmax = atoi(arg[iarg+1]); // max before merge
      npmx = atof(arg[iarg+2]); // max per group
      nGono = atof(arg[iarg+3]); // numbe of praticles to reduce to
      if (atoi(arg[iarg+4]) <= 0) x2flag = 0;
      else x2flag = 1;
      if (atoi(arg[iarg+5]) <= 0) v2flag = 0;
      else v2flag = 1;
      if (atoi(arg[iarg+6]) <= 0) v3flag = 0;
      else v3flag = 1;
      nm = 10; // by default, conserves mass, CoM, momentum, diag. of second raw moment
      if(x2flag > 0) nm += 3; // second spatial moment for variance
      if(v2flag > 0) nm += 3; // off-diag. second raw moment for shear stress
      if(v3flag > 0) nm += 3; // third raw velocity moment for heat flux
      if(nGono <= nm) error->one(FLERR,"Number reduced must be greater than number of constraints");
      iarg += 7;
    } else if (strcmp(arg[iarg],"position") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal collide_modify command");
      if (strcmp(arg[iarg+1],"rnd") == 0) rFlag = 0;
      else if (strcmp(arg[iarg+1],"lws") == 0) rFlag = 1;
      else if (strcmp(arg[iarg+1],"mean") == 0) rFlag = 2;
      else if (strcmp(arg[iarg+1],"dev") == 0) rFlag = 3;
      else error->all(FLERR,"Illegal collide_modify command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"gthresh") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal collide_modify command");
      if (strcmp(arg[iarg+1],"gn") == 0) gthFlag = 0;
      else if (strcmp(arg[iarg+1],"n") == 0) gthFlag = 1;
      else error->all(FLERR,"Illegal collide_modify command");
      iarg += 2;
    } else error->all(FLERR,"Illegal collide_modify command");
  }
}

/* ----------------------------------------------------------------------
   reset vremax to initial species-based values
   reset remain to 0.0
------------------------------------------------------------------------- */

void Collide::reset_vremax()
{
  for (int icell = 0; icell < nglocal; icell++)
    for (int igroup = 0; igroup < ngroups; igroup++)
      for (int jgroup = 0; jgroup < ngroups; jgroup++) {
        vremax[icell][igroup][jgroup] = vremax_initial[igroup][jgroup];
        if (remainflag) remain[icell][igroup][jgroup] = 0.0;
      }
}

/* ----------------------------------------------------------------------
  NTC algorithm
------------------------------------------------------------------------- */

void Collide::collisions()
{
  // if requested, reset vrwmax & remain

  if (update->ntimestep == vre_next) {
    reset_vremax();
    vre_next += vre_every;
  }

  // counters

  ncollide_one = nattempt_one = nreact_one = 0;
  ndelete = 0;

  // perform collisions:
  // variant for single group or multiple groups
  // variant for nearcp flag or not
  //ariant for ambipolar approximation or not
  // variant for stochastic weights
  if (swpmflag) {
    collisions_one_sw<0>(); // nearest neighbor not currently supported
    if(Nmax > 0) {
      if(reduceMinFlag < 0) sw_reduce();
      else sw_reduce_dev();
    }
  } else if (!ambiflag) {
    if (nearcp == 0) {
      if (ngroups == 1) collisions_one<0>();
      else collisions_group<0>();
    } else {
      if (ngroups == 1) collisions_one<1>();
      else collisions_group<1>();
    }
  } else {
    if (ngroups == 1) collisions_one_ambipolar();
    else collisions_group_ambipolar();
  }

  // remove any particles deleted in chemistry reactions
  // if reactions occurred, particles are no longer sorted
  // e.g. compress_reactions may have reallocated particle->next vector
  if (ndelete) particle->compress_reactions(ndelete,dellist);
  if (react) particle->sorted = 0;
  if (swpmflag) particle->sorted = 0;

  // accumulate running totals
  nattempt_running += nattempt_one;
  ncollide_running += ncollide_one;
  nreact_running += nreact_one;
}

/* ----------------------------------------------------------------------
   NTC algorithm for a single group
------------------------------------------------------------------------- */

template < int NEARCP > void Collide::collisions_one()
{
  int i,j,k,m,n,ip,np;
  int nattempt,reactflag;
  double attempt,volume;
  Particle::OnePart *ipart,*jpart,*kpart;

  // loop over cells I own

  Grid::ChildInfo *cinfo = grid->cinfo;

  Particle::OnePart *particles = particle->particles;
  int *next = particle->next;

  for (int icell = 0; icell < nglocal; icell++) {
    np = cinfo[icell].count;
    if (np <= 1) continue;

    if (NEARCP) {
      if (np > max_nn) realloc_nn(np,nn_last_partner);
      memset(nn_last_partner,0,np*sizeof(int));
    }

    ip = cinfo[icell].first;
    volume = cinfo[icell].volume / cinfo[icell].weight;
    if (volume == 0.0) error->one(FLERR,"Collision cell volume is zero");

    // setup particle list for this cell

    if (np > npmax) {
      while (np > npmax) npmax += DELTAPART;
      memory->destroy(plist);
      memory->create(plist,npmax,"collide:plist");
    }

    n = 0;
    while (ip >= 0) {
      plist[n++] = ip;
      ip = next[ip];
    }

    // attempt = exact collision attempt count for all particles in cell
    // nattempt = rounded attempt with RN
    // if no attempts, continue to next grid cell

    attempt = attempt_collision(icell,np,volume);
    nattempt = static_cast<int> (attempt);

    if (!nattempt) continue;
    nattempt_one += nattempt;

    // perform collisions
    // select random pair of particles, cannot be same
    // test if collision actually occurs
    for (int iattempt = 0; iattempt < nattempt; iattempt++) {
      i = np * random->uniform();
      if (NEARCP) j = find_nn(i,np);
      else {
        j = np * random->uniform();
        while (i == j) j = np * random->uniform();
      }

      ipart = &particles[plist[i]];
      jpart = &particles[plist[j]];

      // test if collision actually occurs
      // continue to next collision if no reaction

      if (!test_collision(icell,0,0,ipart,jpart)) continue;

      if (NEARCP) {
        nn_last_partner[i] = j+1;
        nn_last_partner[j] = i+1;
      }

      // if recombination reaction is possible for this IJ pair
      // pick a 3rd particle to participate and set cell number density
      // unless boost factor turns it off, or there is no 3rd particle

      if (recombflag && recomb_ijflag[ipart->ispecies][jpart->ispecies]) {
        if (random->uniform() > react->recomb_boost_inverse)
          react->recomb_species = -1;
        else if (np <= 2)
          react->recomb_species = -1;
        else {
          k = np * random->uniform();
          while (k == i || k == j) k = np * random->uniform();
          react->recomb_part3 = &particles[plist[k]];
          react->recomb_species = react->recomb_part3->ispecies;
          react->recomb_density = np * update->fnum / volume;
        }
      }

      // perform collision and possible reaction

      setup_collision(ipart,jpart);
      reactflag = perform_collision(ipart,jpart,kpart);
      ncollide_one++;
      if (reactflag) nreact_one++;
      else continue;

      // if jpart destroyed: delete from plist, add particle to deletion list
      // exit attempt loop if only single particle left

      if (!jpart) {
        if (ndelete == maxdelete) {
          maxdelete += DELTADELETE;
          memory->grow(dellist,maxdelete,"collide:dellist");
        }
        dellist[ndelete++] = plist[j];
        np--;
        plist[j] = plist[np];
        if (NEARCP) nn_last_partner[j] = nn_last_partner[np];
        if (np < 2) break;
      }

      // if kpart created, add to plist
      // kpart was just added to particle list, so index = nlocal-1
      // particle data structs may have been realloced by kpart

      if (kpart) {
        if (np == npmax) {
          npmax += DELTAPART;
          memory->grow(plist,npmax,"collide:plist");
        }
        if (NEARCP) set_nn(np);
        plist[np++] = particle->nlocal-1;
        particles = particle->particles;
      }
    } // end attempt loop
  } // end cell loop
}

/* ----------------------------------------------------------------------
   NTC algorithm for multiple groups, loop over pairs of groups
   pre-compute # of attempts per group pair
------------------------------------------------------------------------- */

template < int NEARCP > void Collide::collisions_group()
{
  int i,j,k,m,n,ii,jj,kk,ip,np,isp,ng;
  int pindex,ipair,igroup,jgroup,newgroup,ngmax;
  int nattempt,reactflag;
  int *ni,*nj,*ilist,*jlist;
  int *nn_igroup,*nn_jgroup;
  double attempt,volume;
  Particle::OnePart *ipart,*jpart,*kpart;

  // loop over cells I own

  Grid::ChildInfo *cinfo = grid->cinfo;

  Particle::OnePart *particles = particle->particles;
  int *next = particle->next;
  int *species2group = mixture->species2group;

  for (int icell = 0; icell < nglocal; icell++) {
    np = cinfo[icell].count;
    if (np <= 1) continue;
    ip = cinfo[icell].first;
    volume = cinfo[icell].volume / cinfo[icell].weight;
    if (volume == 0.0) error->one(FLERR,"Collision cell volume is zero");

    // reallocate plist and p2g if necessary

    if (np > npmax) {
      while (np > npmax) npmax += DELTAPART;
      memory->destroy(plist);
      memory->create(plist,npmax,"collide:plist");
      memory->destroy(p2g);
      memory->create(p2g,npmax,2,"collide:p2g");
    }

    // plist = particle list for entire cell
    // glist[igroup][i] = index in plist of Ith particle in Igroup
    // ngroup[igroup] = particle count in Igroup
    // p2g[i][0] = Igroup for Ith particle in plist
    // p2g[i][1] = index within glist[igroup] of Ith particle in plist

    for (i = 0; i < ngroups; i++) ngroup[i] = 0;
    n = 0;

    while (ip >= 0) {
      isp = particles[ip].ispecies;
      igroup = species2group[isp];
      if (ngroup[igroup] == maxgroup[igroup]) {
        maxgroup[igroup] += DELTAPART;
        memory->grow(glist[igroup],maxgroup[igroup],"collide:glist");
      }
      ng = ngroup[igroup];
      glist[igroup][ng] = n;
      p2g[n][0] = igroup;
      p2g[n][1] = ng;
      plist[n] = ip;
      ngroup[igroup]++;
      n++;
      ip = next[ip];
    }

    if (NEARCP) {
      ngmax = 0;
      for (i = 0; i < ngroups; i++) ngmax = MAX(ngmax,ngroup[i]);
      if (ngmax > max_nn) {
        realloc_nn(ngmax,nn_last_partner_igroup);
        realloc_nn(ngmax,nn_last_partner_jgroup);
      }
    }

    // attempt = exact collision attempt count for a pair of groups
    // double loop over N^2 / 2 pairs of groups
    // nattempt = rounded attempt with RN
    // NOTE: not using RN for rounding of nattempt
    // gpair = list of group pairs when nattempt > 0

    npair = 0;
    for (igroup = 0; igroup < ngroups; igroup++)
      for (jgroup = igroup; jgroup < ngroups; jgroup++) {
        attempt = attempt_collision(icell,igroup,jgroup,volume);
        nattempt = static_cast<int> (attempt);

        if (nattempt) {
          gpair[npair][0] = igroup;
          gpair[npair][1] = jgroup;
          gpair[npair][2] = nattempt;
          nattempt_one += nattempt;
          npair++;
        }
      }

    // perform collisions for each pair of groups in gpair list
    // select random particle in each group
    // if igroup = jgroup, cannot be same particle
    // test if collision actually occurs
    // if chemistry occurs, move output I,J,K particles to new group lists
    // if chemistry occurs, exit attempt loop if group counts become too small
    // Ni and Nj are pointers to value in ngroup vector
    //   b/c need to stay current as chemistry occurs
    // NOTE: OK to use pre-computed nattempt when Ngroup may have changed via react?

    for (ipair = 0; ipair < npair; ipair++) {
      igroup = gpair[ipair][0];
      jgroup = gpair[ipair][1];
      nattempt = gpair[ipair][2];

      ni = &ngroup[igroup];
      nj = &ngroup[jgroup];
      ilist = glist[igroup];
      jlist = glist[jgroup];

      // re-test for no possible attempts
      // could have changed due to reactions in previous group pairs

      if (*ni == 0 || *nj == 0) continue;
      if (igroup == jgroup && *ni == 1) continue;

      if (NEARCP) {
        nn_igroup = nn_last_partner_igroup;
        if (igroup == jgroup) nn_jgroup = nn_last_partner_igroup;
        else nn_jgroup = nn_last_partner_jgroup;
        memset(nn_igroup,0,(*ni)*sizeof(int));
        if (igroup != jgroup) memset(nn_jgroup,0,(*nj)*sizeof(int));
      }

      for (int iattempt = 0; iattempt < nattempt; iattempt++) {
        i = *ni * random->uniform();
        if (NEARCP) j = find_nn_group(i,ilist,*nj,jlist,plist,nn_igroup,nn_jgroup);
        else {
          j = *nj * random->uniform();
          if (igroup == jgroup)
            while (i == j) j = *nj * random->uniform();
        }

        ipart = &particles[plist[ilist[i]]];
        jpart = &particles[plist[jlist[j]]];

        // test if collision actually occurs
        // continue to next collision if no reaction

        if (!test_collision(icell,igroup,jgroup,ipart,jpart)) continue;

        if (NEARCP) {
          nn_igroup[i] = j+1;
          nn_jgroup[j] = i+1;
        }

        // if recombination reaction is possible for this IJ pair
        // pick a 3rd particle to participate and set cell number density
        // unless boost factor turns it off, or there is no 3rd particle

        if (recombflag && recomb_ijflag[ipart->ispecies][jpart->ispecies]) {
          if (random->uniform() > react->recomb_boost_inverse)
            react->recomb_species = -1;
          else if (np <= 2)
            react->recomb_species = -1;
          else {
            ii = ilist[i];
            jj = jlist[j];
            k = np * random->uniform();
            while (k == ii || k == jj) k = np * random->uniform();
            react->recomb_part3 = &particles[plist[k]];
            react->recomb_species = react->recomb_part3->ispecies;
            react->recomb_density = np * update->fnum / volume;
          }
        }

        // perform collision and possible reaction

        setup_collision(ipart,jpart);
        reactflag = perform_collision(ipart,jpart,kpart);
        ncollide_one++;
        if (reactflag) nreact_one++;
        else continue;

        // ipart may now be in different group
        // reset ilist,jlist after addgroup() in case it realloced glist

        newgroup = species2group[ipart->ispecies];
        if (newgroup != igroup) {
          addgroup(newgroup,ilist[i]);
          delgroup(igroup,i);
          ilist = glist[igroup];
          jlist = glist[jgroup];
          // this line needed if jgroup=igroup and delgroup() moved J particle
          if (jgroup == igroup && j == *ni) j = i;
        }

        // jpart may now be in different group or destroyed
        // if new group: reset ilist,jlist after addgroup() in case it realloced glist
        // if destroyed: delete from plist and group, add particle to deletion list

        if (jpart) {
          newgroup = species2group[jpart->ispecies];
          if (newgroup != jgroup) {
            addgroup(newgroup,jlist[j]);
            delgroup(jgroup,j);
            ilist = glist[igroup];
            jlist = glist[jgroup];
          }

        } else {
          if (ndelete == maxdelete) {
            maxdelete += DELTADELETE;
            memory->grow(dellist,maxdelete,"collide:dellist");
          }
          pindex = jlist[j];
          dellist[ndelete++] = plist[pindex];

          delgroup(jgroup,j);

          plist[pindex] = plist[np-1];
          p2g[pindex][0] = p2g[np-1][0];
          p2g[pindex][1] = p2g[np-1][1];
          if (pindex < np-1) glist[p2g[pindex][0]][p2g[pindex][1]] = pindex;
          np--;

          if (NEARCP) nn_jgroup[j] = nn_jgroup[*nj];
        }

        // if kpart created, add to plist and group list
        // kpart was just added to particle list, so index = nlocal-1
        // reset ilist,jlist after addgroup() in case it realloced
        // particles data struct may also have been realloced

        if (kpart) {
          newgroup = species2group[kpart->ispecies];

          if (NEARCP) {
            if (newgroup == igroup || newgroup == jgroup) {
              n = ngroup[newgroup];
              set_nn_group(n);
              nn_igroup = nn_last_partner_igroup;
              if (igroup == jgroup) nn_jgroup = nn_last_partner_igroup;
              else nn_jgroup = nn_last_partner_jgroup;
              nn_igroup[n] = 0;
              nn_jgroup[n] = 0;
            }
          }

          if (np == npmax) {
            npmax += DELTAPART;
            memory->grow(plist,npmax,"collide:plist");
            memory->grow(p2g,npmax,2,"collide:p2g");
          }
          plist[np++] = particle->nlocal-1;

          addgroup(newgroup,np-1);
          ilist = glist[igroup];
          jlist = glist[jgroup];
          particles = particle->particles;
        }

        // test to exit attempt loop due to groups becoming too small

        if (*ni <= 1) {
          if (*ni == 0) break;
          if (igroup == jgroup) break;
        }
        if (*nj <= 1) {
          if (*nj == 0) break;
          if (igroup == jgroup) break;
        }
      }
    }
  }
}

/* ----------------------------------------------------------------------
   SWPM algorithm for a single group
------------------------------------------------------------------------- */

template < int NEARCP > void Collide::collisions_one_sw()
{
  int i,j,m,n,ip,np;
  int nattempt,reactflag;
  double attempt,volume;

  double c1, c2, pacc, pmx;
  double gi, gj;
  double gsum, gavg; // probabiltiy of selecting particle

  Particle::OnePart *ipart,*jpart,*kpart,*lpart;
  // loop over cells I own
  /* Note: Near Neighbor may not be suitable for SWPM
           due to different weights. Needs to be checked.
  */
  Grid::ChildInfo *cinfo = grid->cinfo;

  Particle::OnePart *particles = particle->particles;
  int *next = particle->next;

  for (int icell = 0; icell < nglocal; icell++) {
    np = cinfo[icell].count;
    if (np <= 1) continue;

    ip = cinfo[icell].first;
    volume = cinfo[icell].volume / cinfo[icell].weight;
    if (volume == 0.0) error->one(FLERR,"Collision cell volume is zero");

    // setup particle list for this cell
    if (np > npmax) {
      while (np > npmax) npmax += DELTAPART;
      memory->destroy(plist);
      memory->create(plist,npmax,"collide:plist");
    }

    // Merge before collisions
    // create particle list
    // AND calculate total weight, max weight, and min weight for selections and probability
    // gmn should be assumed to be zero in case more than one collision occurs where particle
    // ... weight becomes smaller than gmn
    n = 0;
    gsum = gmx = gmn = 0.0;
    while (ip >= 0) {
      plist[n++] = ip;

      ipart = &particles[ip];
      gi = ipart->sw;
      gsum += gi;
      if(n == 1) {
        //gmn = gi;
        gmx = gi;
      } else {
        //gmn = MIN(gmn,gi);
        gmx = MAX(gmx,gi);
      }
      if(gi != gi) error->one(FLERR,"nan weight");
      if(gi < 0) error->one(FLERR,"negative weight");
      ip = next[ip];
    }
    if(gmn < 0) error->one(FLERR,"negative weight");

    if(np < Nmin || Nmin < 0) gamFlag = 1;
    else gamFlag = -1;

    //attempt = attempt_collision_sum(icell,np,volume,gsum);
    attempt = attempt_collision_max(icell,np,volume);

    if(attempt < 0) error->one(FLERR,"negative attempts");
    nattempt = static_cast<int> (attempt);
    if(nattempt != nattempt) error->one(FLERR,"attempt error");
    if(nattempt < 0) {
      printf("attempt: %4.3e; nattempt: %i\n", attempt, nattempt);
      error->one(FLERR,"negative nattempts");
    }

    if (!nattempt) continue;
    nattempt_one += nattempt;

    for(int iattempt = 0; iattempt < nattempt; iattempt++) {
      /*if(np == 2) {
        i = 0;
        ipart = &particles[plist[i]];
        j = 1;
        jpart = &particles[plist[j]];
      } else {
        // assume gmin = 0
        // use accept-reject
        // cf 3.96 + 3.97 stochastic numerics BE
        pacc = 0;
        pmx = ((np - 2)*gmx + gsum)/(2*(np - 1)*gsum); // expedites selection
        while(pacc/pmx < random->uniform()) {
          i = np * random->uniform();
          ipart = &particles[plist[i]];
          gi = ipart->sw;
          pacc = ((np - 2)*gi + gsum)/(2*(np - 1)*gsum); // c1 = gsum
        }

        pacc = 0;
        j = i;
        pmx = (gi + gmx)/((np-2)*gi + gsum); // expedites selection
        int itry = 0;
        while(pacc/pmx < random->uniform() || i==j) {
          j = np * random->uniform();
          while(i == j) j = np * random->uniform();
          jpart = &particles[plist[j]];
          gj = jpart->sw;
          pacc = (gi + gj)/((np-2)*gi + gsum); // c1 = gi
          if(itry>np) break;
          itry++;
        }

        // probabilities similar since all very small
        // randomly choose
        if(itry>np) {
          j = i;
          while(i==j) j = np * random->uniform();
          jpart = &particles[plist[j]];
        }
      }*/

      i = np * random->uniform();
      j = np * random->uniform();
      while (i == j) j = np * random->uniform();
      ipart = &particles[plist[i]];
      jpart = &particles[plist[j]];

      //if (!test_collision_sum(icell,0,0,ipart,jpart)) continue;
      if (!test_collision_max(icell,0,0,ipart,jpart)) continue;

      // +2 (split)
      if(gamFlag > 0) {
        perform_split(ipart,jpart,kpart,lpart);

        // Add new particles kpart and lpart to plist
        if(kpart) {
          perform_collision_sw(kpart,lpart);
          if (np+2 >= npmax) {
            npmax += DELTAPART;
            memory->grow(plist,npmax,"collide:plist");
          }
          // note two were added
          plist[np++] = particle->nlocal-2;
          plist[np++] = particle->nlocal-1;          
          particles = particle->particles;
        } else {
          perform_collision_sw(ipart,jpart);
        }
      // +1/0
      } else {
        perform_collision_sw(ipart,jpart,kpart);
        if (kpart) {
          if (np >= npmax) {
            npmax += DELTAPART;
            memory->grow(plist,npmax,"collide:plist");
          }
          plist[np++] = particle->nlocal-1;
          particles = particle->particles;
        }
      } // conditional for collision / splits

      ncollide_one++;
    } // loop for attempts

		// Manually remove small weighted particles
    gavg = gsum/np;
    double gremove = 0.0;
    int nremain = np;
    for(i = 0; i < np; i++) {
      ipart = &particles[plist[i]];
      gi = ipart->sw;
      // if weight is too small
      if(gi < gavg/1e6 || gi < 1) {
        gremove += gi;
        nremain--;
        if (ndelete == maxdelete) {
          maxdelete += DELTADELETE;
          memory->grow(dellist,maxdelete,"collide:dellist");
        }
        ipart->sw = -1;
        dellist[ndelete++] = plist[i];
      }
    }
    // assume the total removed weight is << remaining weight
    double gsum2 = 0.0;
    double galloc = gremove/static_cast <double> (nremain);
    for(i = 0; i < np; i++) {
      ipart = &particles[plist[i]];
      gi = ipart->sw;
      // if weight is too small
      if(gi > 0) {
        ipart->sw = gi + galloc;
        gsum2 += ipart->sw;
      }
    }
    if(gremove/gsum > 0.001) {
      printf("gsum-gsum2: %4.8e; gremove: %4.8e\n", gsum-gsum2, gremove);
      printf("gremain: %4.8e; gremove/gorig: %4.8e\n", gsum-gremove, gremove/gsum);
      printf("many removed\n");
      if(nremain == 0) error->one(FLERR,"all removed");
    }

  }// loop for cells
}

/* ----------------------------------------------------------------------
   Reduction: Deletes particles cell by cell. Needed for multiple reductions
   in case one reduction does not reduce np<Nmax.
------------------------------------------------------------------------- */

void Collide::sw_reduce()
{
  int i,j,m,n,ip,np;
  int nattempt,reactflag;
  double attempt,volume;
  Particle::OnePart *ipart;
  Grid::ChildInfo *cinfo = grid->cinfo;
  Particle::OnePart *particles = particle->particles;
  int *next = particle->next;

  //int redFlag = 0;
  for (int icell = 0; icell < nglocal; icell++) {
    cinfo[icell].ndel = 0;
    // create particle list
    ip = cinfo[icell].first;
    n = 0;
    while (ip >= 0) {
      plist[n++] = ip;
      ip = next[ip];
    }
    if (n <= Nmax) continue;

    double Abuf = 2.0;
    while(n > Nmax) {
      // create particle list
      ip = cinfo[icell].first;
      n = 0;
      double gmean = 0.0;
      while (ip >= 0) {
        ipart = &particles[ip];
        double g = ipart->sw;

        // remove negative weight particles from plist
        if(g > 0) {
          plist[n++] = ip;
          double d1 = g - gmean;
          gmean += (d1/m);
        }
        ip = next[ip];
      }
      gthresh = Abuf*gmean*npThresh;

      // reduce more if the number of particles is still too many
      // AND if the reduction is still reducing
      dnRed = 0;
      xmr[0] = xmr[1] = xmr[2] = 0.0;
      divideMerge(plist,n);
      n -= dnRed;

      cinfo[icell].ndel += dnRed;

      // if none reduced, increase weight threshold
      // if too big, break
      // should have to use this scarcely
      if(dnRed == 0) Abuf += 2.0;
      if(Abuf > 20) break; // avoid infinite loop
    }
    //redFlag = 1;
  }// loop for cells
}

/* ----------------------------------------------------------------------
   Same as sw_reduce() but partitions particles in space first
------------------------------------------------------------------------- */

void Collide::sw_reduce_dev()
{
  int i,j,m,n,p,ip,np;
  int nattempt,reactflag;
  double attempt,volume;
  Particle::OnePart *ipart;
  Grid::ChildInfo *cinfo = grid->cinfo;
  Particle::OnePart *particles = particle->particles;
  int *next = particle->next;
  Grid::ChildCell *cells = grid->cells;

  //int redFlag = 0;
  //double xl[3], xh[3], xp[3];
  //double subdx, subdy, subdz;
  int halfn;
  double gmean, gvar, gstd;
  for (int icell = 0; icell < nglocal; icell++) {
    cinfo[icell].ndel = 0;
    // create particle list
    ip = cinfo[icell].first;
    n = 0;
    while (ip >= 0) {
      plist[n++] = ip;
      ip = next[ip];
    }
    if (n <= Nmax) continue;

    // there should not be so many collisions for this to be called more than once
    double Abuf = 2.0;
    while(n > Nmax) {
      // find mean weight
      ip = cinfo[icell].first;
      n = 0;
      gmean = gvar = 0.0;
      while (ip >= 0) {
        ipart = &particles[ip];
        double g = ipart->sw;

        // Incremental variance
        double d1, d2;
        if(g > 0) {
          n++;
          d1 = g - gmean;
          gmean += (d1/n);
          gvar += (n-1.0)/n*d1*d1;
        }
        ip = next[ip];
      }
      gstd = sqrt(gvar/n);

      // Would like all reduced particles to have gmean as weight
      // Set weight bound based on number reduced
      // buffer for grouping
      gthresh = Abuf*gmean*npThresh;

      dnRed = 0;
      // merge outliers
      ip = cinfo[icell].first;
      n = 0;
      while (ip >= 0) {
        ipart = &particles[ip];
        double g = ipart->sw;
        if(g > 0 && g <= (gmean-1.65*gstd)) plist[n++] = ip;
        ip = next[ip];
      }
      xmr[0] = xmr[1] = xmr[2] = 0.0;
      divideMerge(plist,n);

      // last, merge all but higher outliers
      // high outliers are where particle weight exceed 2x mean weight
      ip = cinfo[icell].first;
      n = 0;
      while (ip >= 0) {
        ipart = &particles[ip];
        double g = ipart->sw;
        if(g > 0 && g <= 2.0*gmean) plist[n++] = ip;
        ip = next[ip];
      }
      xmr[0] = xmr[1] = xmr[2] = 0.0;
      divideMerge(plist,n);

      n -= dnRed;
      cinfo[icell].ndel += dnRed;

      // if none reduced, increase weight threshold
      // if too big, break
      // should have to use this scarcely
      if(dnRed == 0) Abuf += 2.0;
      if(Abuf > 20) break;
    }
    //redFlag = 1;
  }// loop for cells
}

/* ----------------------------------------------------------------------
   NTC algorithm for a single group with ambipolar approximation
------------------------------------------------------------------------- */

void Collide::collisions_one_ambipolar()
{
  int i,j,k,n,ip,np,nelectron,nptotal,ispecies,jspecies,tmp;
  int nattempt,reactflag;
  double attempt,volume;
  Particle::OnePart *ipart,*jpart,*kpart,*p,*ep;

  // ambipolar vectors

  int *ionambi = particle->eivec[particle->ewhich[index_ionambi]];
  double **velambi = particle->edarray[particle->ewhich[index_velambi]];

  // loop over cells I own

  Grid::ChildInfo *cinfo = grid->cinfo;

  Particle::OnePart *particles = particle->particles;
  int *next = particle->next;
  int nbytes = sizeof(Particle::OnePart);

  for (int icell = 0; icell < nglocal; icell++) {
    np = cinfo[icell].count;
    if (np <= 1) continue;
    ip = cinfo[icell].first;
    volume = cinfo[icell].volume / cinfo[icell].weight;
    if (volume == 0.0) error->one(FLERR,"Collision cell volume is zero");

    // setup particle list for this cell

    if (np > npmax) {
      while (np > npmax) npmax += DELTAPART;
      memory->destroy(plist);
      memory->create(plist,npmax,"collide:plist");
    }

    n = 0;
    while (ip >= 0) {
      plist[n++] = ip;
      ip = next[ip];
    }

    // setup elist of ionized electrons for this cell
    // create them in separate array since will never become real particles

    if (np >= maxelectron) {
      while (maxelectron < np) maxelectron += DELTAELECTRON;
      memory->sfree(elist);
      elist = (Particle::OnePart *)
        memory->smalloc(maxelectron*nbytes,"collide:elist");
    }

    // create electrons for ambipolar ions

    nelectron = 0;
    for (i = 0; i < np; i++) {
      if (ionambi[plist[i]]) {
        p = &particles[plist[i]];
        ep = &elist[nelectron];
        memcpy(ep,p,nbytes);
        memcpy(ep->v,velambi[plist[i]],3*sizeof(double));
        ep->ispecies = ambispecies;
        nelectron++;
      }
    }

    // attempt = exact collision attempt count for all particles in cell
    // nptotal = includes neutrals, ions, electrons
    // nattempt = rounded attempt with RN

    nptotal = np + nelectron;
    attempt = attempt_collision(icell,nptotal,volume);
    nattempt = static_cast<int> (attempt);

    if (!nattempt) continue;
    nattempt_one += nattempt;

    // perform collisions
    // select random pair of particles, cannot be same
    // test if collision actually occurs
    // if chemistry occurs, exit attempt loop if group count goes to 0

    for (int iattempt = 0; iattempt < nattempt; iattempt++) {
      i = nptotal * random->uniform();
      j = nptotal * random->uniform();
      while (i == j) j = nptotal * random->uniform();

      // ipart,jpart = heavy particles or electrons

      if (i < np) ipart = &particles[plist[i]];
      else ipart = &elist[i-np];
      if (j < np) jpart = &particles[plist[j]];
      else jpart = &elist[j-np];

      // check for e/e pair
      // count as collision, but do not perform it

      if (ipart->ispecies == ambispecies && jpart->ispecies == ambispecies) {
        ncollide_one++;
        continue;
      }

      // if particle I is electron
      // swap with J, since electron must be 2nd in any ambipolar reaction
      // just need to swap i/j, ipart/jpart
      // don't have to worry if an ambipolar ion is I or J

      if (ipart->ispecies == ambispecies) {
        tmp = i;
        i = j;
        j = tmp;
        p = ipart;
        ipart = jpart;
        jpart = p;
      }

      // test if collision actually occurs

      if (!test_collision(icell,0,0,ipart,jpart)) continue;

      // if recombination reaction is possible for this IJ pair
      // pick a 3rd particle to participate and set cell number density
      // unless boost factor turns it off, or there is no 3rd particle
      // 3rd particle cannot be an electron, so select from Np

      if (recombflag && recomb_ijflag[ipart->ispecies][jpart->ispecies]) {
        if (random->uniform() > react->recomb_boost_inverse)
          react->recomb_species = -1;
        else if (np == 1)
          react->recomb_species = -1;
        else if (np == 2 && jpart->ispecies != ambispecies)
          react->recomb_species = -1;
        else {
          k = np * random->uniform();
          while (k == i || k == j) k = np * random->uniform();
          react->recomb_part3 = &particles[plist[k]];
          react->recomb_species = react->recomb_part3->ispecies;
          react->recomb_density = np * update->fnum / volume;
        }
      }

      // perform collision
      // ijspecies = species before collision chemistry
      // continue to next collision if no reaction

      ispecies = ipart->ispecies;
      jspecies = jpart->ispecies;
      setup_collision(ipart,jpart);
      reactflag = perform_collision(ipart,jpart,kpart);
      ncollide_one++;
      if (reactflag) nreact_one++;
      else continue;

      // reset ambipolar ion flags due to collision
      // must do now before particle count reset below can break out of loop
      // first reset ionambi if kpart was added since ambi_reset() uses it

      if (kpart) ionambi = particle->eivec[particle->ewhich[index_ionambi]];
      if (jspecies == ambispecies)
        ambi_reset(plist[i],-1,jspecies,ipart,jpart,kpart,ionambi);
      else
        ambi_reset(plist[i],plist[j],jspecies,ipart,jpart,kpart,ionambi);

      // if kpart created:
      // particles and custom data structs may have been realloced by kpart
      // add kpart to plist or elist
      // kpart was just added to particle list, so index = nlocal-1
      // must come before jpart code below since it modifies nlocal

      if (kpart) {
        particles = particle->particles;
        ionambi = particle->eivec[particle->ewhich[index_ionambi]];
        velambi = particle->edarray[particle->ewhich[index_velambi]];

        if (kpart->ispecies != ambispecies) {
          if (np == npmax) {
            npmax += DELTAPART;
            memory->grow(plist,npmax,"collide:plist");
          }
          plist[np++] = particle->nlocal-1;

        } else {
          if (nelectron == maxelectron) {
            maxelectron += DELTAELECTRON;
            elist = (Particle::OnePart *)
              memory->srealloc(elist,maxelectron*nbytes,"collide:elist");
          }
          ep = &elist[nelectron];
          memcpy(ep,kpart,nbytes);
          ep->ispecies = ambispecies;
          nelectron++;
          particle->nlocal--;
        }
      }

      // if jpart exists, was originally not an electron, now is an electron:
      //   ionization reaction converted 2 neutrals to one ion
      //   add to elist, remove from plist, flag J for deletion
      // if jpart exists, was originally an electron, now is not an electron:
      //   exchange reaction converted ion + electron to two neutrals
      //   add neutral J to master particle list, remove from elist, add to plist
      // if jpart destroyed, was an electron:
      //   recombination reaction converted ion + electron to one neutral
      //   remove electron from elist
      // else if jpart destroyed:
      //   non-ambipolar recombination reaction
      //   remove from plist, flag J for deletion

      if (jpart) {
        if (jspecies != ambispecies && jpart->ispecies == ambispecies) {
          if (nelectron == maxelectron) {
            maxelectron += DELTAELECTRON;
            elist = (Particle::OnePart *)
              memory->srealloc(elist,maxelectron*nbytes,"collide:elist");
          }
          ep = &elist[nelectron];
          memcpy(ep,jpart,nbytes);
          ep->ispecies = ambispecies;
          nelectron++;
          jpart = NULL;

        } else if (jspecies == ambispecies && jpart->ispecies != ambispecies) {
          int reallocflag = particle->add_particle();
          if (reallocflag) {
            particles = particle->particles;
            ionambi = particle->eivec[particle->ewhich[index_ionambi]];
            velambi = particle->edarray[particle->ewhich[index_velambi]];
          }

          int index = particle->nlocal-1;
          memcpy(&particles[index],jpart,nbytes);
          particles[index].id = MAXSMALLINT*random->uniform();
          ionambi[index] = 0;

          if (nelectron-1 != j-np) memcpy(&elist[j-np],&elist[nelectron-1],nbytes);
          nelectron--;

          if (np == npmax) {
            npmax += DELTAPART;
            memory->grow(plist,npmax,"collide:plist");
          }
          plist[np++] = index;
        }
      }

      if (!jpart && jspecies == ambispecies) {
        if (nelectron-1 != j-np) memcpy(&elist[j-np],&elist[nelectron-1],nbytes);
        nelectron--;

      } else if (!jpart) {
        if (ndelete == maxdelete) {
          maxdelete += DELTADELETE;
          memory->grow(dellist,maxdelete,"collide:dellist");
        }
        dellist[ndelete++] = plist[j];
        plist[j] = plist[np-1];
        np--;
      }

      // update particle counts
      // quit if no longer enough particles for another collision

      nptotal = np + nelectron;
      if (nptotal < 2) break;
    }

    // done with collisions/chemistry for one grid cell
    // recombine ambipolar ions with their matching electrons
    //   by copying electron velocity into velambi
    // which ion is combined with which electron does not matter
    // error if ion count does not match electron count

    int melectron = 0;
    for (n = 0; n < np; n++) {
      i = plist[n];
      if (ionambi[i]) {
        if (melectron < nelectron) {
          ep = &elist[melectron];
          memcpy(velambi[i],ep->v,3*sizeof(double));
        }
        melectron++;
      }
    }
    if (melectron != nelectron)
      error->one(FLERR,"Collisions in cell did not conserve electron count");
  }
}

/* ----------------------------------------------------------------------
   NTC algorithm for multiple groups with ambipolar approximation
   loop over pairs of groups, pre-compute # of attempts per group pair
------------------------------------------------------------------------- */

void Collide::collisions_group_ambipolar()
{
  int i,j,k,n,ii,jj,ip,np,isp,ng;
  int pindex,ipair,igroup,jgroup,newgroup,ispecies,jspecies,tmp;
  int nattempt,reactflag,nelectron;
  int *ni,*nj,*ilist,*jlist,*tmpvec;
  double attempt,volume;
  Particle::OnePart *ipart,*jpart,*kpart,*p,*ep;

  // ambipolar vectors

  int *ionambi = particle->eivec[particle->ewhich[index_ionambi]];
  double **velambi = particle->edarray[particle->ewhich[index_velambi]];

  // loop over cells I own

  Grid::ChildInfo *cinfo = grid->cinfo;

  Particle::OnePart *particles = particle->particles;
  int *next = particle->next;
  int nbytes = sizeof(Particle::OnePart);
  int *species2group = mixture->species2group;
  int egroup = species2group[ambispecies];

  for (int icell = 0; icell < nglocal; icell++) {
    np = cinfo[icell].count;
    if (np <= 1) continue;
    ip = cinfo[icell].first;
    volume = cinfo[icell].volume / cinfo[icell].weight;
    if (volume == 0.0) error->one(FLERR,"Collision cell volume is zero");

    // reallocate plist and p2g if necessary

    if (np > npmax) {
      while (np > npmax) npmax += DELTAPART;
      memory->destroy(plist);
      memory->create(plist,npmax,"collide:plist");
      memory->destroy(p2g);
      memory->create(p2g,npmax,2,"collide:p2g");
    }

    // setup elist of ionized electrons for this cell
    // create them in separate array since will never become real particles

    if (np >= maxelectron) {
      while (maxelectron < np) maxelectron += DELTAELECTRON;
      memory->sfree(elist);
      elist = (Particle::OnePart *)
        memory->smalloc(maxelectron*nbytes,"collide:elist");
    }

    // plist = particle list for entire cell
    // glist[igroup][i] = index in plist of Ith particle in Igroup
    // ngroup[igroup] = particle count in Igroup
    // p2g[i][0] = Igroup for Ith particle in plist
    // p2g[i][1] = index within glist[igroup] of Ith particle in plist
    // also populate elist with ionized electrons, now separated from ions
    // ngroup[egroup] = nelectron

    for (i = 0; i < ngroups; i++) ngroup[i] = 0;
    n = 0;
    nelectron = 0;

    while (ip >= 0) {
      isp = particles[ip].ispecies;
      igroup = species2group[isp];
      if (ngroup[igroup] == maxgroup[igroup]) {
        maxgroup[igroup] += DELTAPART;
        memory->grow(glist[igroup],maxgroup[igroup],"collide:glist");
      }
      ng = ngroup[igroup];
      glist[igroup][ng] = n;
      p2g[n][0] = igroup;
      p2g[n][1] = ng;
      plist[n] = ip;
      ngroup[igroup]++;

      if (ionambi[ip]) {
        p = &particles[ip];
        ep = &elist[nelectron];
        memcpy(ep,p,nbytes);
        memcpy(ep->v,velambi[ip],3*sizeof(double));
        ep->ispecies = ambispecies;
        nelectron++;

        if (ngroup[egroup] == maxgroup[egroup]) {
          maxgroup[egroup] += DELTAPART;
          memory->grow(glist[egroup],maxgroup[egroup],"collide:grouplist");
        }
        ng = ngroup[egroup];
        glist[egroup][ng] = nelectron-1;
        ngroup[egroup]++;
      }

      n++;
      ip = next[ip];
    }

    // attempt = exact collision attempt count for a pair of groups
    // double loop over N^2 / 2 pairs of groups
    // temporarily include nelectrons in count for egroup
    // nattempt = rounded attempt with RN
    // NOTE: not using RN for rounding of nattempt
    // gpair = list of group pairs when nattempt > 0
    //         flip igroup/jgroup if igroup = egroup
    // egroup/egroup collisions are not included in gpair

    npair = 0;
    for (igroup = 0; igroup < ngroups; igroup++)
      for (jgroup = igroup; jgroup < ngroups; jgroup++) {
        if (igroup == egroup && jgroup == egroup) continue;
        attempt = attempt_collision(icell,igroup,jgroup,volume);
        nattempt = static_cast<int> (attempt);

        if (nattempt) {
          if (igroup == egroup) {
              gpair[npair][0] = jgroup;
              gpair[npair][1] = igroup;
            } else {
              gpair[npair][0] = igroup;
              gpair[npair][1] = jgroup;
            }
          gpair[npair][2] = nattempt;
          nattempt_one += nattempt;
          npair++;
        }
      }

    // perform collisions for each pair of groups in gpair list
    // select random particle in each group
    // if igroup = jgroup, cannot be same particle
    // test if collision actually occurs
    // if chemistry occurs, move output I,J,K particles to new group lists
    // if chemistry occurs, exit attempt loop if group counts become too small
    // Ni and Nj are pointers to value in ngroup vector
    //   b/c need to stay current as chemistry occurs
    // NOTE: OK to use pre-computed nattempt when Ngroup may have changed via react?

    for (ipair = 0; ipair < npair; ipair++) {
      igroup = gpair[ipair][0];
      jgroup = gpair[ipair][1];
      nattempt = gpair[ipair][2];

      ni = &ngroup[igroup];
      nj = &ngroup[jgroup];
      ilist = glist[igroup];
      jlist = glist[jgroup];

      // re-test for no possible attempts
      // could have changed due to reactions in previous group pairs

      if (*ni == 0 || *nj == 0) continue;
      if (igroup == jgroup && *ni == 1) continue;

      for (int iattempt = 0; iattempt < nattempt; iattempt++) {
        i = *ni * random->uniform();
        j = *nj * random->uniform();
        if (igroup == jgroup)
          while (i == j) j = *nj * random->uniform();

        // ipart/jpart can be from particles or elist

        if (igroup == egroup) ipart = &elist[i];
        else ipart = &particles[plist[ilist[i]]];
        if (jgroup == egroup) jpart = &elist[j];
        else jpart = &particles[plist[jlist[j]]];

        // NOTE: unlike single group, no possibility of e/e collision
        //       means collision stats may be different

        //if (ipart->ispecies == ambispecies && jpart->ispecies == ambispecies) {
        //  ncollide_one++;
        //  continue;
        //}

        // test if collision actually occurs

        if (!test_collision(icell,igroup,jgroup,ipart,jpart)) continue;

        // if recombination reaction is possible for this IJ pair
        // pick a 3rd particle to participate and set cell number density
        // unless boost factor turns it off, or there is no 3rd particle
        // 3rd particle will never be an electron since plist has no electrons
        // if jgroup == egroup, no need to check k for match to jj

        if (recombflag && recomb_ijflag[ipart->ispecies][jpart->ispecies]) {
          if (random->uniform() > react->recomb_boost_inverse)
            react->recomb_species = -1;
          else if (np <= 2)
            react->recomb_species = -1;
          else {
            ii = ilist[i];
            if (jgroup == egroup) jj = -1;
            else jj = jlist[j];
            k = np * random->uniform();
            while (k == ii || k == jj) k = np * random->uniform();
            react->recomb_part3 = &particles[plist[k]];
            react->recomb_species = react->recomb_part3->ispecies;
            react->recomb_density = np * update->fnum / volume;
          }
        }

        // perform collision
        // ijspecies = species before collision chemistry
        // continue to next collision if no reaction

        ispecies = ipart->ispecies;
        jspecies = jpart->ispecies;
        setup_collision(ipart,jpart);
        reactflag = perform_collision(ipart,jpart,kpart);
        ncollide_one++;
        if (reactflag) nreact_one++;
        else continue;

        // reset ambipolar ion flags due to reaction
        // must do now before group reset below can break out of loop
        // first reset ionambi if kpart was added since ambi_reset() uses it

        if (kpart) ionambi = particle->eivec[particle->ewhich[index_ionambi]];
        if (jgroup == egroup)
          ambi_reset(plist[ilist[i]],-1,jspecies,ipart,jpart,kpart,ionambi);
        else
          ambi_reset(plist[ilist[i]],plist[jlist[j]],jspecies,
                     ipart,jpart,kpart,ionambi);

        // ipart may now be in different group
        // reset ilist,jlist after addgroup() in case it realloced glist

        newgroup = species2group[ipart->ispecies];
        if (newgroup != igroup) {
          addgroup(newgroup,ilist[i]);
          delgroup(igroup,i);
          ilist = glist[igroup];
          jlist = glist[jgroup];
          // this line needed if jgroup=igroup and delgroup() moved J particle
          if (jlist == ilist && j == *ni) j = i;
        }

        // if kpart created:
        // particles and custom data structs may have been realloced by kpart
        // add kpart to plist or elist and to group
        // kpart was just added to particle list, so index = nlocal-1
        // must come before jpart code below since it modifies nlocal

        if (kpart) {
          particles = particle->particles;
          ionambi = particle->eivec[particle->ewhich[index_ionambi]];
          velambi = particle->edarray[particle->ewhich[index_velambi]];

          newgroup = species2group[kpart->ispecies];

          if (newgroup != egroup) {
            if (np == npmax) {
              npmax += DELTAPART;
              memory->grow(plist,npmax,"collide:plist");
              memory->grow(p2g,npmax,2,"collide:p2g");
            }
            plist[np++] = particle->nlocal-1;
            addgroup(newgroup,np-1);
            ilist = glist[igroup];
            jlist = glist[jgroup];

          } else {
            if (nelectron == maxelectron) {
              maxelectron += DELTAELECTRON;
              elist = (Particle::OnePart *)
                memory->srealloc(elist,maxelectron*nbytes,"collide:elist");
            }
            ep = &elist[nelectron];
            memcpy(ep,kpart,nbytes);
            ep->ispecies = ambispecies;
            nelectron++;
            particle->nlocal--;

            if (ngroup[egroup] == maxgroup[egroup]) {
              maxgroup[egroup] += DELTAPART;
              memory->grow(glist[egroup],maxgroup[egroup],"collide:grouplist");
            }
            ng = ngroup[egroup];
            glist[egroup][ng] = nelectron-1;
            ngroup[egroup]++;
          }
        }

        // jpart may now be in a different group or destroyed
        // if jpart exists, now in a different group, neither group is egroup:
        //   add/del group, reset ilist,jlist after addgroup() in case glist realloced
        // if jpart exists, was originally not an electron, now is an electron:
        //   ionization reaction converted 2 neutrals to one ion
        //   add to elist, remove from plist, flag J for deletion
        // if jpart exists, was originally an electron, now is not an electron:
        //   exchange reaction converted ion + electron to two neutrals
        //   add neutral J to master particle list, remove from elist, add to plist
        // if jpart destroyed, was an electron:
        //   recombination reaction converted ion + electron to one neutral
        //   remove electron from elist
        // else if jpart destroyed:
        //   non-ambipolar recombination reaction
        //   remove from plist and group, add particle to deletion list

        if (jpart) {
          newgroup = species2group[jpart->ispecies];

          if (newgroup == jgroup) {
            // nothing to do

          } else if (jgroup != egroup && newgroup != egroup) {
            addgroup(newgroup,jlist[j]);
            delgroup(jgroup,j);
            ilist = glist[igroup];
            jlist = glist[jgroup];

          } else if (jgroup != egroup && jpart->ispecies == ambispecies) {
            if (nelectron == maxelectron) {
              maxelectron += DELTAELECTRON;
              elist = (Particle::OnePart *)
                memory->srealloc(elist,maxelectron*nbytes,"collide:elist");
            }
            ep = &elist[nelectron];
            memcpy(ep,jpart,nbytes);
            ep->ispecies = ambispecies;
            nelectron++;

            if (ngroup[egroup] == maxgroup[egroup]) {
              maxgroup[egroup] += DELTAPART;
              memory->grow(glist[egroup],maxgroup[egroup],"collide:grouplist");
            }
            ng = ngroup[egroup];
            glist[egroup][ng] = nelectron-1;
            ngroup[egroup]++;

            jpart = NULL;

          } else if (jgroup == egroup && jpart->ispecies != ambispecies) {
            int reallocflag = particle->add_particle();
            if (reallocflag) {
              particles = particle->particles;
              ionambi = particle->eivec[particle->ewhich[index_ionambi]];
              velambi = particle->edarray[particle->ewhich[index_velambi]];
            }

            int index = particle->nlocal-1;
            memcpy(&particles[index],jpart,nbytes);
            particles[index].id = MAXSMALLINT*random->uniform();
            ionambi[index] = 0;

            if (nelectron-1 != j) memcpy(&elist[j],&elist[nelectron-1],nbytes);
            nelectron--;
            ngroup[egroup]--;

            if (np == npmax) {
              npmax += DELTAPART;
              memory->grow(plist,npmax,"collide:plist");
              memory->grow(p2g,npmax,2,"collide:p2g");
            }
            plist[np++] = index;
            addgroup(newgroup,np-1);
            ilist = glist[igroup];
            jlist = glist[jgroup];
          }
        }

        if (!jpart && jspecies == ambispecies) {
          if (nelectron-1 != j) memcpy(&elist[j],&elist[nelectron-1],nbytes);
          nelectron--;
          ngroup[egroup]--;

        } else if (!jpart) {
          if (ndelete == maxdelete) {
            maxdelete += DELTADELETE;
            memory->grow(dellist,maxdelete,"collide:dellist");
          }
          pindex = jlist[j];
          dellist[ndelete++] = plist[pindex];

          delgroup(jgroup,j);

          plist[pindex] = plist[np-1];
          p2g[pindex][0] = p2g[np-1][0];
          p2g[pindex][1] = p2g[np-1][1];
          if (pindex < np-1) glist[p2g[pindex][0]][p2g[pindex][1]] = pindex;
          np--;
        }

        // test to exit attempt loop due to groups becoming too small

        if (*ni <= 1) {
          if (*ni == 0) break;
          if (igroup == jgroup) break;
        }
        if (*nj <= 1) {
          if (*nj == 0) break;
          if (igroup == jgroup) break;
        }
      }
    }

    // done with collisions/chemistry for one grid cell
    // recombine ambipolar ions with their matching electrons
    //   by copying electron velocity into velambi
    // which ion is combined with which electron does not matter
    // error if do not use all nelectrons in cell

    int melectron = 0;
    for (n = 0; n < np; n++) {
      i = plist[n];
      if (ionambi[i]) {
        if (melectron < nelectron) {
          ep = &elist[melectron];
          memcpy(velambi[i],ep->v,3*sizeof(double));
        }
        melectron++;
      }
    }
    if (melectron != nelectron)
      error->one(FLERR,"Collisions in cell did not conserve electron count");
  }
}

/* ----------------------------------------------------------------------
   reset ionambi flags if ambipolar reaction occurred
   this operates independent of cell particle counts and plist/elist data structs
     caller will adjust those after this method returns
   i/j = indices of I,J reactants
   isp/jsp = pre-reaction species of I,J
     both will not be electrons, if one is electron it will be jsp
   reactants i,j and isp/jsp will always be in order listed below
   products ip,jp,kp will always be in order listed below
   logic must be valid for all ambipolar AND non-ambipolar reactions
   check for 3 versions of 2 -> 3: dissociation or ionization
     all have J product = electron
     D: AB + e -> A + e + B
        if I reactant = neutral and K product not electron:
        set K product = neutral
     D: AB+ + e -> A+ + e + B
        if I reactant = ion:
        set K product = neutral
     I: A + e -> A+ + e + e
        if I reactant = neutral and K product = electron:
        set I product = ion
     all other 2 -> 3 cases, set K product = neutral
   check for 4 versions of 2 -> 2: ionization or exchange
     I: A + B -> AB+ + e
        if J product = electron:
        set I product to ion
     E: AB+ + e -> A + B
        if I reactant = ion and J reactant = elecrton
        set I/J products to neutral
     E: AB+ + C -> A + BC+
        if I reactant = ion:
        set I/J products to neutral/ion
     E: C + AB+ -> A + BC+
        if J reactant = ion:
        nothing to change for products
     all other 2 -> 2 cases, no changes
   check for one version of 2 -> 1: recombination
     R: A+ + e -> A
        if ej = elec, set I product to neutral
     all other 2 -> 1 cases, no changes
   WARNING:
     do not index by I,J if could be e, since may be negative I,J index
     do not access ionambi if could be e, since e may be in elist
------------------------------------------------------------------------- */

void Collide::ambi_reset(int i, int j, int jsp,
                         Particle::OnePart *ip, Particle::OnePart *jp,
                         Particle::OnePart *kp, int *ionambi)
{
  int e = ambispecies;

  // 2 reactants become 3 products
  // in all ambi reactions with an electron reactant, it is J

  if (kp) {
    int k = particle->nlocal-1;
    ionambi[k] = 0;
    if (jsp != e) return;

    if (ionambi[i]) {                // nothing to change
    } else if (kp->ispecies == e) {
      ionambi[i] = 1;                // 1st reactant is now 1st product ion
    }

  // 2 reactants become 2 products
  // ambi reaction if J product is electron or either reactant is ion

  } else if (jp) {
    if (jp->ispecies == e) {
      ionambi[i] = 1;         // 1st reactant is now 1st product ion
    } else if (ionambi[i] && jsp == e) {
      ionambi[i] = 0;         // 1st reactant is now 1st product neutral
    } else if (ionambi[i]) {
      ionambi[i] = 0;         // 1st reactant is now 1st product neutral
      ionambi[j] = 1;         // 2nd reactant is now 2nd product ion
    }

  // 2 reactants become 1 product
  // ambi reaction if J reactant is electron

  } else if (!jp) {
    if (jsp == e) ionambi[i] = 0;   // 1st reactant is now 1st product neutral
  }
}

/* ----------------------------------------------------------------------
   pack icell values for per-cell arrays into buf
   if icell is a split cell, also pack all sub cell values
   return byte count of amount packed
   if memflag, only return count, do not fill buf
   NOTE: why packing/unpacking parent cell if a split cell?
------------------------------------------------------------------------- */

int Collide::pack_grid_one(int icell, char *buf, int memflag)
{
  int nbytes = ngroups*ngroups*sizeof(double);

  Grid::ChildCell *cells = grid->cells;

  int n;
  if (remainflag) {
    if (memflag) {
      memcpy(buf,&vremax[icell][0][0],nbytes);
      memcpy(&buf[nbytes],&remain[icell][0][0],nbytes);
    }
    n = 2*nbytes;
  } else {
    if (memflag) memcpy(buf,&vremax[icell][0][0],nbytes);
    n = nbytes;
  }

  if (cells[icell].nsplit > 1) {
    int isplit = cells[icell].isplit;
    int nsplit = cells[icell].nsplit;
    for (int i = 0; i < nsplit; i++) {
      int m = grid->sinfo[isplit].csubs[i];
      if (remainflag) {
        if (memflag) {
          memcpy(&buf[n],&vremax[m][0][0],nbytes);
          n += nbytes;
          memcpy(&buf[n],&remain[m][0][0],nbytes);
          n += nbytes;
        } else n += 2*nbytes;
      } else {
        if (memflag) memcpy(&buf[n],&vremax[m][0][0],nbytes);
        n += nbytes;
      }
    }
  }

  return n;
}

/* ----------------------------------------------------------------------
   unpack icell values for per-cell arrays from buf
   if icell is a split cell, also unpack all sub cell values
   return byte count of amount unpacked
------------------------------------------------------------------------- */

int Collide::unpack_grid_one(int icell, char *buf)
{
  int nbytes = ngroups*ngroups*sizeof(double);

  Grid::ChildCell *cells = grid->cells;
  Grid::SplitInfo *sinfo = grid->sinfo;

  grow_percell(1);
  memcpy(&vremax[icell][0][0],buf,nbytes);
  int n = nbytes;
  if (remainflag) {
    memcpy(&remain[icell][0][0],&buf[n],nbytes);
    n += nbytes;
  }
  nglocal++;

  if (cells[icell].nsplit > 1) {
    int isplit = cells[icell].isplit;
    int nsplit = cells[icell].nsplit;
    grow_percell(nsplit);
    for (int i = 0; i < nsplit; i++) {
      int m = sinfo[isplit].csubs[i];
      memcpy(&vremax[m][0][0],&buf[n],nbytes);
      n += nbytes;
      if (remainflag) {
        memcpy(&remain[m][0][0],&buf[n],nbytes);
        n += nbytes;
      }
    }
    nglocal += nsplit;
  }

  return n;
}

/* ----------------------------------------------------------------------
   copy per-cell collision info from Icell to Jcell
   called whenever a grid cell is removed from this processor's list
   caller checks that Icell != Jcell
------------------------------------------------------------------------- */

void Collide::copy_grid_one(int icell, int jcell)
{
  int nbytes = ngroups*ngroups*sizeof(double);

  memcpy(&vremax[jcell][0][0],&vremax[icell][0][0],nbytes);
  if (remainflag)
    memcpy(&remain[jcell][0][0],&remain[icell][0][0],nbytes);
}

/* ----------------------------------------------------------------------
   reset final grid cell count after grid cell removals
------------------------------------------------------------------------- */

void Collide::reset_grid_count(int nlocal)
{
  nglocal = nlocal;
}

/* ----------------------------------------------------------------------
   add a grid cell
   called when a grid cell is added to this processor's list
   initialize values to 0.0
------------------------------------------------------------------------- */

void Collide::add_grid_one()
{
  grow_percell(1);

  for (int igroup = 0; igroup < ngroups; igroup++)
    for (int jgroup = 0; jgroup < ngroups; jgroup++) {
      vremax[nglocal][igroup][jgroup] = vremax_initial[igroup][jgroup];
      if (remainflag) remain[nglocal][igroup][jgroup] = 0.0;
    }

  nglocal++;
}

/* ----------------------------------------------------------------------
   reinitialize per-cell arrays due to grid cell adaptation
   count of owned grid cells has changed
   called from adapt_grid
------------------------------------------------------------------------- */

void Collide::adapt_grid()
{
  int nglocal_old = nglocal;
  nglocal = grid->nlocal;

  // reallocate vremax and remain
  // initialize only new added locations
  // this leaves vremax/remain for non-adapted cells the same

  nglocalmax = nglocal;
  memory->grow(vremax,nglocalmax,ngroups,ngroups,"collide:vremax");
  if (remainflag)
    memory->grow(remain,nglocalmax,ngroups,ngroups,"collide:remain");

  for (int icell = nglocal_old; icell < nglocal; icell++)
    for (int igroup = 0; igroup < ngroups; igroup++)
      for (int jgroup = 0; jgroup < ngroups; jgroup++) {
        vremax[icell][igroup][jgroup] = vremax_initial[igroup][jgroup];
        if (remainflag) remain[icell][igroup][jgroup] = 0.0;
      }
}

/* ----------------------------------------------------------------------
   insure per-cell arrays are allocated long enough for N new cells
------------------------------------------------------------------------- */

void Collide::grow_percell(int n)
{
  if (nglocal+n < nglocalmax || !ngroups) return;
  while (nglocal+n >= nglocalmax) nglocalmax += DELTAGRID;
  memory->grow(vremax,nglocalmax,ngroups,ngroups,"collide:vremax");
  if (remainflag)
    memory->grow(remain,nglocalmax,ngroups,ngroups,"collide:remain");
}

/* ----------------------------------------------------------------------
   for particle I, find collision partner J via near neighbor algorithm
   always returns a J neighbor, even if not that near
   near neighbor algorithm:
     check up to nearlimit particles, starting with random particle
     as soon as find one within distance moved by particle I, return it
     else return the closest one found
     also exclude an I,J pair if both most recently collided with each other
   this version is for single group collisions
------------------------------------------------------------------------- */

int Collide::find_nn(int i, int np)
{
  int jneigh;
  double dx,dy,dz,rsq;
  double *xj;

  // if np = 2, just return J = non-I particle
  // np is never < 2

  if (np == 2) return (i+1) % 2;

  Particle::OnePart *ipart,*jpart;
  Particle::OnePart *particles = particle->particles;
  double dt = update->dt;

  // thresh = distance particle I moves in this timestep

  ipart = &particles[plist[i]];
  double *vi = ipart->v;
  double *xi = ipart->x;
  double threshsq =  dt*dt * (vi[0]*vi[0]+vi[1]*vi[1]+vi[2]*vi[2]);
  double minrsq = BIG;

  // nlimit = max # of J candidates to consider

  int nlimit = MIN(nearlimit,np-1);
  int count = 0;

  // pick a random starting J
  // jneigh = collision partner when exit loop
  //   set to initial J as default in case no Nlimit J meets criteria

  int j = np * random->uniform();
  while (i == j) j = np * random->uniform();
  jneigh = j;

  while (count < nlimit) {
    count++;

    // skip this J if I,J last collided with each other

    if (nn_last_partner[i] == j+1 && nn_last_partner[j] == i+1) {
      j++;
      if (j == np) j = 0;
      continue;
    }

    // rsq = squared distance between particles I and J
    // if rsq = 0.0, skip this J
    //   could be I = J, or a cloned J at same position as I
    // if rsq <= threshsq, this J is collision partner
    // if rsq = smallest yet seen, this J is tentative collision partner

    jpart = &particles[plist[j]];
    xj = jpart->x;
    dx = xi[0] - xj[0];
    dy = xi[1] - xj[1];
    dz = xi[2] - xj[2];
    rsq = dx*dx + dy*dy + dz*dz;

    if (rsq > 0.0) {
      if (rsq <= threshsq) {
        jneigh = j;
        break;
      }
      if (rsq < minrsq) {
        minrsq = rsq;
        jneigh = j;
      }
    }

    j++;
    if (j == np) j = 0;
  }

  return jneigh;
}

/* ----------------------------------------------------------------------
   for particle I, find collision partner J via near neighbor algorithm
   always returns a J neighbor, even if not that near
   same near neighbor algorithm as in find_nn()
     looking for J particles in jlist of length Np = ngroup[jgroup]
     ilist = jlist when igroup = jgroup
   this version is for multi group collisions
------------------------------------------------------------------------- */

int Collide::find_nn_group(int i, int *ilist, int np, int *jlist, int *plist,
                           int *nn_igroup, int *nn_jgroup)
{
  int jneigh;
  double dx,dy,dz,rsq;
  double *xj;

  // if ilist = jlist and np = 2, just return J = non-I particle
  // np is never < 2 for ilist = jlist
  // np is never < 1 for ilist != jlist

  if (ilist == jlist && np == 2) return (i+1) % 2;

  Particle::OnePart *ipart,*jpart;
  Particle::OnePart *particles = particle->particles;
  double dt = update->dt;

  // thresh = distance particle I moves in this timestep

  ipart = &particles[plist[ilist[i]]];
  double *vi = ipart->v;
  double *xi = ipart->x;
  double threshsq =  dt*dt * (vi[0]*vi[0]+vi[1]*vi[1]+vi[2]*vi[2]);
  double minrsq = BIG;

  // nlimit = max # of J candidates to consider

  int nlimit = MIN(nearlimit,np-1);
  int count = 0;

  // pick a random starting J
  // jneigh = collision partner when exit loop
  //   set to initial J as default in case no Nlimit J meets criteria

  int j = np * random->uniform();
  if (ilist == jlist)
    while (i == j) j = np * random->uniform();
  jneigh = j;

  while (count < nlimit) {
    count++;

    // skip this J if I,J last collided with each other

    if (nn_igroup[i] == j+1 && nn_jgroup[j] == i+1) {
      j++;
      if (j == np) j = 0;
      continue;
    }

    // rsq = squared distance between particles I and J
    // if rsq = 0.0, skip this J
    //   could be I = J, or a cloned J at same position as I
    // if rsq <= threshsq, this J is collision partner
    // if rsq = smallest yet seen, this J is tentative collision partner

    jpart = &particles[plist[jlist[j]]];
    xj = jpart->x;
    dx = xi[0] - xj[0];
    dy = xi[1] - xj[1];
    dz = xi[2] - xj[2];
    rsq = dx*dx + dy*dy + dz*dz;

    if (rsq > 0.0) {
      if (rsq <= threshsq) {
        jneigh = j;
        break;
      }
      if (rsq < minrsq) {
        minrsq = rsq;
        jneigh = j;
      }
    }

    j++;
    if (j == np) j = 0;
  }

  return jneigh;
}

/* ----------------------------------------------------------------------
   reallocate a nn_last_partner vector to allow for N values
   increase size by multiples of 2x
------------------------------------------------------------------------- */

void Collide::realloc_nn(int n, int *&vec)
{
  while (n > max_nn) max_nn *= 2;
  memory->destroy(vec);
  memory->create(vec,max_nn,"collide:nn_last_partner");
}

/* ----------------------------------------------------------------------
   set nn_last_partner[N] = 0 for newly created particle
   grow the vector if necessary
------------------------------------------------------------------------- */

void Collide::set_nn(int n)
{
  if (n == max_nn) {
    max_nn *= 2;
    memory->grow(nn_last_partner,max_nn,"collide:nn_last_partner");
  }
  nn_last_partner[n] = 0;
}

/* ----------------------------------------------------------------------
   grow the group last partner vectors if necessary
------------------------------------------------------------------------- */

void Collide::set_nn_group(int n)
{
  if (n == max_nn) {
    max_nn *= 2;
    memory->grow(nn_last_partner_igroup,max_nn,"collide:nn_last_partner");
    memory->grow(nn_last_partner_jgroup,max_nn,"collide:nn_last_partner");
  }
}

/* ----------------------------------------------------------------------
   Initialize maximum weight in each cell
------------------------------------------------------------------------- */

void Collide::gmax_init()
{
  //for (int icell = 0; icell < nglocal; icell++)
  //  gmax[icell] = gmax_initial;
  return;
}

/* ----------------------------------------------------------------------
   Divide parent cluster to two children according to Rij
------------------------------------------------------------------------- */
void Collide::divideMerge(int *node_pid, int np)
{
  int pid;
  Particle::OnePart *ip;
  Particle::OnePart *particles = particle->particles;

  if(np<=npThresh)
    return; // ignore empty leaves

/*------------------------------------------------------------------------ */
  // compute covariance

	// Based on Ling 1974, a two-pass calculation of mean seems best
  // M11 = Sum(x_i)/n
	// M21 = M11 + Sum(x_i - M11) / n;
	// first compute M11 with Kahan sum
  Particle::Species *species = particle->species;

  double gsum, gV[3], gVV[3][3], gVVV[3];
  double V11[3], Vsq;
  double mass = species[0].mass;

  gsum = 0.0;
  for(int i = 0; i < 3; i++) {
    gV[i] = 0.0;
    V11[i] = 0.0;
    gVVV[i] = 0.0;
    for(int j = 0; j < 3; j++)
      gVV[i][j] = 0.0;
  }

  // for KahanSum
	// change to Neumaier (improved Kahan sum)
  double cgsum, cgV[3], cgVV[3][3], cgVVV[3];
  cgsum = 0.0;
  for(int i = 0; i < 3; i++) {
    cgV[i] = 0.0;
    cgVVV[i] = 0.0;
    for(int j = 0; j < 3; j++)
      cgVV[i][j] = 0.0;
  }

  // Compute raw moments
  double gp, vp[3], vpsq;
  // Copy weights and velocities into temporary arrays
  double gpA[np], vpA[np][3];
  for(int p = 0; p < np; p++) {
    ip = &particles[node_pid[p]];
    memcpy(vp,ip->v,3*sizeof(double));
    gp = ip->sw;
    gpA[p] = gp;
    for(int d = 0; d < 3; d++) vpA[p][d] = vp[d];
  }

  // sort from smallest to largest weight using insertion sort
  double gpins, vpins[3];
  int r;
  for(int p = 0; p < np; p++) {
    for(int d = 0; d < 3; d++) vpins[d] = vpA[p][d];
    gpins = gpA[p];
    r = p - 1;
    while(r >= 0 && gpA[r] > gpins) {
      gpA[r+1] = gpA[r];
      for(int d = 0; d < 3; d++) vpA[r+1][d] = vpA[r][d];
      r--;
    }
    gpA[r+1] = gpins;
    for(int d = 0; d < 3; d++) vpA[r+1][d] = vpins[d];
  }

  double y;
	double t;
  for(int p = 0; p < np; p++) {
    gp = gpA[p];
    for(int d = 0; d < 3; d++) vp[d] = vpA[p][d];

    if(gp < 0) error->one(FLERR,"negative weight");

		y = gp;
    t = gsum + y;
		if(std::abs(gsum) >= std::abs(y)) {
    	cgsum += ((gsum - t) + y);
		} else {
			cgsum += ((y - t) + gsum);
		}
   	gsum = t;

    vpsq = 0.0;
    for(int i = 0; i < 3; i++) {
			y = gp*vp[i];
      t = gV[i] + y;
			if(std::abs(gV[i]) >= std::abs(y)) {
		  	cgV[i] += ((gV[i] - t) + y);
			} else {
				cgV[i] += ((y - t) + gV[i]);
			}
      gV[i] = t;

      // calculate first trial mean for velocity as rolling mean
      V11[i] += gp/gsum*(vp[i]-V11[i]);

      vpsq += vp[i]*vp[i];
      // stress tensor (2nd)
      for(int j = 0; j < 3; j++) { // only need upper triangle
				y = gp*vp[i]*vp[j];
		    t = gVV[i][j] + y;
				if(std::abs(gVV[i][j]) >= std::abs(y)) {
					cgVV[i][j] += ((gVV[i][j] - t) + y);
				} else {
					cgVV[i][j] += ((y - t) + gVV[i][j]);
				}
		    gVV[i][j] = t;
      } // end j
    } // end i

    // heat flux (3rd)
    for(int i = 0; i < 3; i++) {
			y = gp*vp[i]*vpsq;
      t = gVVV[i] + y;
			if(std::abs(gVVV[i]) >= std::abs(y)) {
		  	cgVVV[i] += ((gVVV[i] - t) + y);
			} else {
				cgVVV[i] += ((y - t) + gVVV[i]);
			}
      gVVV[i] = t;
    }
  }

	// add accumulated error at end
	// also avoid asosciativity rule in compiler
  gsum += cgsum;
  for(int i = 0; i < 3; i++) {
    gV[i] += cgV[i];
    gVVV[i] += cgVVV[i];
    for(int j = 0; j < 3; j++)
      gVV[i][j] += cgVV[i][j];
  }

  for(int i = 0; i < 3; i++) gVVV[i] *= 0.5;

	// second pass for mean velocity
	double V[3];
	V[0] = V[1] = V[2] = 0.0;
  for(int p = 0; p < np; p++) {
    gp = gpA[p];
    for(int d = 0; d < 3; d++) vp[d] = vpA[p][d];

    if(gp < 0) error->one(FLERR,"negative weight");
		// calculate second trial mean
    for(int i = 0; i < 3; i++) V[i] += gp/gsum*(vp[i]-V11[i]);
  }
	for(int i = 0; i < 3; i++) V[i] += V11[i];

  // Compute center of mass velocity and its square
  Vsq = 0.0;
  for(int i = 0; i < 3; i++) Vsq += V[i]*V[i];

  // Compute central stress
  double pij[3][3], q[3];
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++) {
      pij[i][j] = gVV[i][j] - gsum*V[i]*V[j];
      Rij[i][j] = pij[i][j]/gsum; 
    } // end j
  } // end i

/*------------------------------------------------------------------------ */
  int reduceFlag = -1;
  if(gthFlag == 0) {
    if(np <= npmx || gsum < gthresh) reduceFlag = 1;
  } else if(gthFlag == 1) {
    if(np <= npmx) reduceFlag = 1;
  } else {
    error->one(FLERR,"no threshold set");
  }

  //if(np < npmx || gsum < gthresh) { // npmx is now the maximum number of particles in a group
  //if(gsum < gthresh) { // npmx is now the maximum number of particles in a group
  if(reduceFlag > 0) {
    // grow currentCluster if needed
    if (np > maxCluster) {
      while (np > maxCluster) maxCluster += 10;
      memory->destroy(currentCluster);
      memory->create(currentCluster,maxCluster,"collide:currentCluster");
    }

    npCluster = np;
    for(int i = 0; i < np; i++) currentCluster[i] = node_pid[i];

    // Temperature and energy
    double T = (pij[0][0] + pij[1][1] + pij[2][2])/(3.0 * gsum);

    if(T < 0 || T != T) {
      for(int i = 0; i < np; i++) {
        ip = &particles[node_pid[i]];
        printf("g: %4.3e; v: %4.3e, %4.3e, %4.3e\n", ip->sw,
          ip->v[0], ip->v[1], ip->v[2]);
      }
      error->one(FLERR,"negative/nan temp");
    }

    // Compute central heat flux
    for(int i = 0; i < 3; i++) q[i] = gVVV[i] -
      (pij[i][0]*V[0] + pij[i][1]*V[1] + pij[i][2]*V[2]) -
      0.5*gsum*V[i]*Vsq - 1.5*gsum*T*V[i];

    // Store variables
    iT = sqrt(3.0*T);
    igsum = gsum;
    for(int i = 0; i < 3; i++) {
      ie[i] = gVV[i][i];
      iV[i] = V[i];
      iq[i] = q[i];
      for(int j = 0; j < 3; j++) {
        ipij[i][j] = pij[i][j];
      } // end j
    } // end i

    merge();
    return;
  }

  // Each column is an eigenvector
  int ierror = MathEigen::jacobi3(Rij,eval,evec);

  // Find largest eigenpair
  // First eigen pair since its sort=SORT_DECREASING_EVALS and
  // covariance matrices have non-negative eigenvalues
  double maxeval;
  double maxevec[3]; // normal of splitting plane

  maxeval = 0;
  for (int i = 0; i < 3; i++) {
    if(std::abs(eval[i])>maxeval) { // eigenvalues all positive
      maxeval = std::abs(eval[i]);
      for (int j = 0; j < 3; j++) {
        maxevec[j] = evec[j][i];  
      }
    }
  }

  // sort and then separate
  double center = MathExtra::dot3(V,maxevec);
  int pidL[np];
  int pidR[np];
  int npL, npR;
  npL = npR = 0;
  for(int i = 0; i < np; i++) {
    pid = node_pid[i];
    ip = &particles[pid];
    if(MathExtra::dot3(ip->v,maxevec) < center)
      pidL[npL++] = pid;
    else
      pidR[npR++] = pid;
  }

  if(npL == np || npR == np || npL == 0 || npR == 0) {
    printf("V: %4.3e, %4.3e, %4.3e\n", V[0], V[1], V[2]);
    for(int i = 0; i < np; i++) {
      pid = node_pid[i];
      ip = &particles[pid];
      printf("pid: %i; gp: %4.3e; vp: %4.3e, %4.3e, %4.3e\n", pid, ip->sw, ip->v[0], ip->v[1], ip->v[2]);
    } 
    printf("maxevec: %4.3e, %4.3e, %4.3e\n", maxevec[0], maxevec[1], maxevec[2]);
    error->one(FLERR,"no change after division");
  }

  divideMerge(pidL,npL);
  divideMerge(pidR,npR);
}

/* ----------------------------------------------------------------------
   Merge particles in each cluster
------------------------------------------------------------------------- */
void Collide::merge()
{
  int i,j,k,l,m,n; // for loop vars
  int ip,jp,kp,lp,mp,np; // particle ind vars
  double *ixp, *ivp, igp;
  double *jxp, *jvp, jgp;
  Particle::OnePart *ipart, *jpart;
  Particle::OnePart *kpart, *lpart, *mpart, *npart;
  Particle::OnePart *particles = particle->particles;

  Grid::ChildCell *cells = grid->cells;

  //////////////////////////////////////////////
  // Center of mass / geometric median
  //////////////////////////////////////////////
  double gX[3];
  gX[0] = gX[1] = gX[2] = 0.0;
  double gXX[3];
  gXX[0] = gXX[1] = gXX[2] = 0.0;
  double xm[3];
  if(rFlag >= 2) {
    for(i = 0; i < npCluster; i++) {
      ipart = &particles[currentCluster[i]];
      ixp = ipart->x;
      igp = ipart->sw;
      for(int d = 0; d < 3; d++) {

        gX[d] += (igp*ixp[d]);
        gXX[d] += (igp*ixp[d]*ixp[d]);
      }
    } // end np loop
    for(int d = 0; d < 3; d++) xm[d] = gX[d] / igsum;
  }

  //////////////////////////////////////////////
  // Center of mass / geometric median
  //////////////////////////////////////////////
  double xl[3], xh[3];
  int icell = particles[currentCluster[0]].icell;
  for(int d = 0; d < 3; d++) {
    xl[d] = cells[icell].lo[d];
    xh[d] = cells[icell].hi[d];
  }

  //////////////////////////////////////////////
  // Reduce particles
  //////////////////////////////////////////////
  if(reduction_type == 0) { // Conserve energy
    double uvec[3];
    double theta = 2.0 * 3.14159 * random->uniform();
    double phi = acos(1.0 - 2.0 * random->uniform());
    uvec[0] = sin(phi) * cos(theta);
    uvec[1] = sin(phi) * sin(theta);
    uvec[2] = cos(phi);
    

    // determine new velocities
    double vi[3], vj[3];
    for(int d = 0; d < 3; d++) {
      vi[d] = iV[d] + iT*uvec[d];
      vj[d] = iV[d] - iT*uvec[d];
    }

    // determine new weights
    double gi = igsum * 0.5;
    double gj = igsum * 0.5;

    // select particle pair with bias toward larger weight
    if(rFlag == 1) {
      error->one(FLERR,"unstable atm");
    // choose pair whose CoM is closest to original
    } else if (rFlag == 3) {
      double xp[npCluster][3];
      for(i = 0; i < npCluster; i++) {
        ipart = &particles[currentCluster[i]];
        for(int d = 0; d < 3; d++) xp[i][d] = ipart->x[d];
      }

      // carry over CoM difference
      double dgX[3];
      dgX[0] = dgX[1] = dgX[2] = 0.0;
      for(int d = 0; d < 3; d++) dgX[d] = xmr[d];

      double mxmag = 0.0;
      double dxmag = 0.0;
      double dxm[3], mxm[3];
      for(i = 0; i < npCluster; i++) {
        for(j = 0; j < npCluster; j++) {
          if(i == j) continue;
          dxmag = 0.0;
          for(int d = 0; d < 3; d++) {
            dxm[d] = (gi*xp[i][d] + gj*xp[j][d]) - (gX[d] + dgX[d]);
            dxmag += dxm[d]*dxm[d];
          }
          if((i==0 && j==1) || dxmag < mxmag) {
            ip = i;
            jp = j;
            mxmag = dxmag;
            for(int d = 0; d < 3; d++) mxm[d] = dxm[d];
          }
        }
      }

      // store difference in center of mass for next iteration
      for(int d = 0; d < 3; d++) xmr[d] = mxm[d];      

      ipart = &particles[currentCluster[ip]];
      jpart = &particles[currentCluster[jp]];

    // randomly select pair
    } else {
      ip = random->uniform()*npCluster;
      jp = random->uniform()*npCluster;
      while(ip==jp) jp = random->uniform()*npCluster;
      ipart = &particles[currentCluster[ip]];
      jpart = &particles[currentCluster[jp]];
    }

    ipart->sw = gi;
    jpart->sw = gj;

    for(int d = 0; d < 3; d++) {
      ipart->v[d] = vi[d];
      jpart->v[d] = vj[d];

      if(rFlag == 2) {
        ipart->x[d] = xm[d];
        jpart->x[d] = xm[d];
      }
    }

    // delete other particles
    for(int idel = 0; idel < npCluster; idel++) {
      if(idel==ip || idel==jp) continue;
      if (ndelete == maxdelete) {
        maxdelete += DELTADELETE;
        memory->grow(dellist,maxdelete,"collide:dellist");
      }
      ipart = &particles[currentCluster[idel]];
      ipart->sw = -1; // needed to do multple merge cycles
      dellist[ndelete++] = currentCluster[idel];
      dnRed++;
    }

  // Rjasanow, Schreiber, and Wagner (1997) merge
  } else if (reduction_type == 1) {
    // checked moments properly conserved
    // Precalculate
    double alpha, beta, itheta;
    double uvec[3]; // center of mass velocity and unit vector
    double qmag = sqrt(iq[0]*iq[0] + iq[1]*iq[1] + iq[2]*iq[2]);
    double qge;
    if(qmag<SMALLNUM) {
      qmag = 0.0;
      itheta = 1.0;
      alpha = iT;
      beta = iT;
      for (int d = 0; d < 3; d++) {
        double A = sqrt(-log(random->uniform()));
        double theta = 6.283185308 * random->uniform();
        if(random->uniform()<0.5) uvec[d] = A * cos(theta);
        else uvec[d] = A * sin(theta);
      }
    } else {
      for(int d = 0; d < 3; d++) uvec[d] = iq[d]/qmag;
      double qge = qmag / (igsum * pow(iT,3.0));
      if(qge<=0.0) qge = 0.0;
      itheta = qge + sqrt(1.0 + qge * qge);
      alpha = itheta*iT; 
      beta = iT/itheta;
    }

    // determine new velocities
    double vi[3], vj[3];
    for(int d = 0; d < 3; d++) {
      vi[d] = iV[d] + alpha*uvec[d];
      vj[d] = iV[d] - beta*uvec[d];
    }

    // determine new weights
    double gi = igsum / (1.0 + itheta * itheta);
    double gj = igsum - gi;

		if(gi != gi || gj != gj || gi < 0 || gj < 0)
			error->one(FLERR,"invalid weight");

    // select particle pair with bias toward larger weight
    if(rFlag == 1) {
      error->one(FLERR,"unstable atm");
      /*double gp;
      ip = random->uniform()*npCluster;
      ipart = &particles[currentCluster[ip]];
      gp = ipart->sw;
      while(gp/igsum < random->uniform()) {
        ip = random->uniform()*npCluster;
        ipart = &particles[currentCluster[ip]];
        gp = ipart->sw;
      } 

      jp = random->uniform()*npCluster;
      jpart = &particles[currentCluster[jp]];
      gp = jpart->sw;
      while(gp/igsum < random->uniform() || ip==jp) {
        jp = random->uniform()*npCluster;
        jpart = &particles[currentCluster[jp]];
        gp = jpart->sw;
      }*/
    // choose pair whose CoM is closest to original
    } else if (rFlag == 3) {
      double xp[npCluster][3];
      for(i = 0; i < npCluster; i++) {
        ipart = &particles[currentCluster[i]];
        for(int d = 0; d < 3; d++) xp[i][d] = ipart->x[d];
      }

      // carry over CoM difference
      double dgX[3];
      dgX[0] = dgX[1] = dgX[2] = 0.0;
      for(int d = 0; d < 3; d++) dgX[d] = xmr[d];

      double mxmag = 0.0;
      double dxmag = 0.0;
      double dxm[3], mxm[3];
      for(i = 0; i < npCluster; i++) {
        for(j = 0; j < npCluster; j++) {
          if(i == j) continue;
          dxmag = 0.0;
          for(int d = 0; d < 3; d++) {
            dxm[d] = (gi*xp[i][d] + gj*xp[j][d]) - (gX[d] + dgX[d]);
            dxmag += dxm[d]*dxm[d];
          }
          if((i==0 && j==1) || dxmag < mxmag) {
            ip = i;
            jp = j;
            mxmag = dxmag;
            for(int d = 0; d < 3; d++) mxm[d] = dxm[d];
          }
        }
      }

      // store difference in center of mass for next iteration
      for(int d = 0; d < 3; d++) xmr[d] = mxm[d];      

      ipart = &particles[currentCluster[ip]];
      jpart = &particles[currentCluster[jp]];

    // randomly select pair
    } else {
      ip = random->uniform()*npCluster;
      jp = random->uniform()*npCluster;
      while(ip==jp) jp = random->uniform()*npCluster;
      ipart = &particles[currentCluster[ip]];
      jpart = &particles[currentCluster[jp]];
    }

    ipart->sw = gi;
    jpart->sw = gj;

    if(gi != gi) {
      printf("gi: %4.3e\n", gi);
      printf("iq: %4.3e, %4.3e, %4.3e\n", iq[0], iq[1], iq[2]);
      printf("uv: %4.3e, %4.3e, %4.3e\n", uvec[0], uvec[1], uvec[2]);
      printf("iT: %4.3e\n", iT);
      printf("rho: %4.3e; itheta: %4.3e\n", igsum, itheta);
      error->one(FLERR,"weight is wrong");
    }
    if(gi != gi) {
      printf("gj: %4.3e\n", gj);
      printf("iq: %4.3e, %4.3e, %4.3e\n", iq[0], iq[1], iq[2]);
      printf("uv: %4.3e, %4.3e, %4.3e\n", uvec[0], uvec[1], uvec[2]);
      printf("iT: %4.3e\n", iT);
      printf("rho: %4.3e; itheta: %4.3e\n", igsum, itheta);
      error->one(FLERR,"weight is wrong");
    }

    for(int d = 0; d < 3; d++) {
      ipart->v[d] = vi[d];
      jpart->v[d] = vj[d];

      if(rFlag == 2) {
        ipart->x[d] = xm[d];
        jpart->x[d] = xm[d];
      }
    }

    // delete other particles
    for(int idel = 0; idel < npCluster; idel++) {
      if(idel==ip || idel==jp) continue;
      if(ndelete == maxdelete) {
        maxdelete += DELTADELETE;
        memory->grow(dellist,maxdelete,"collide:dellist");
      }
      ipart = &particles[currentCluster[idel]];
      ipart->sw = -1; // needed to do multple merge cycles
      dellist[ndelete++] = currentCluster[idel];
      dnRed++;
    }

  // LGZ (2020)
  } else if (reduction_type == 2) {
    // checked moments properly conserved (not responsible for small weight)

    // Find eigenpairs for the stress tensor
    int ierror = MathEigen::jacobi3(ipij,eval,evec); // double check eigenvec

    // Find number of nonzero eigenpair
    int nK = 0;
    double ieval[3], ievec[3][3];
    for (i = 0; i < 3; i++) {
      if( std::abs(eval[i]) >= SMALLNUM) {
        ieval[nK] = eval[i];
        for(int d = 0; d < 3; d++) ievec[nK][d] = evec[i][d];
        nK++;
      }
    } // end i

    double qli;
    double rhoklam, qrhoklam, gamma[3];
    double orhoklam;
    double gi[3], gj[3];
    gi[0] = gi[1] = gi[2] = 0.0;
    gj[0] = gj[1] = gj[2] = 0.0;

    double vi[3][3], vj[3][3];
    // max nK can be is 3 so all vectors/matrices large enough
    for(int iK = 0; iK < nK; iK++) {
      //if(signK[iK] == 0) continue;

      // determine direction of eigenvector
      qli = ievec[0][iK]*iq[0] + ievec[1][iK]*iq[1] + ievec[2][iK]*iq[2];
      if(qli<0)
        for(int d = 0; d < 3; d++) ievec[d][iK] *= -1.0;
      qli = std::abs(qli);

      // precompute remaining constants
      gamma[iK] = sqrt(igsum) * qli / (sqrt(nK) * pow(ieval[iK],1.5))
        + sqrt(1.0 + (igsum*qli*qli)/(nK*pow(ieval[iK],3.0)));

      gi[iK] = igsum / (nK * (1.0 + gamma[iK] * gamma[iK]));
      gj[iK] = igsum * gamma[iK] * gamma[iK] / (nK * (1.0 + gamma[iK] * gamma[iK]));

			if(gi[iK] != gi[iK] || gj[iK] != gj[iK] || gi[iK] < 0 || gj[iK] < 0)
				error->one(FLERR,"invalid weight");

      // set new velocities and weights (double checked)
      for(int d = 0; d < 3; d++) {
        vi[iK][d] = iV[d] + gamma[iK]*sqrt(nK*ieval[iK]/igsum)*ievec[d][iK];
        vj[iK][d] = iV[d] - 1.0/gamma[iK]*sqrt(nK*ieval[iK]/igsum)*ievec[d][iK];
      }
    }

    // Stochastically swap to order the particles by weight
    if(rFlag == 1) { // may stall and also does not solve any issues
      error->one(FLERR,"unstable atm");
    // redistribute weight between pairs to exactly conserve CoM
    } else if (rFlag == 3) {
      // randomly choose 6 points by shuffling list
      for(i = npCluster-1; i > 0; i--) {
        j = random->uniform()*(i+1);
        std::swap(currentCluster[i], currentCluster[j]);
      }

      // copy particle positions into temporary array
      double txp[6][3];
      for(i = 0; i < 6; i++) {
        ipart = &particles[currentCluster[i]];
        for(int d = 0; d < 3; d++) txp[i][d] = ipart->x[d]; 
      }

      // calculate center of masses of each pair
      double xmp[3][3]; //[x,y,z] [particle pair]
      int iK = 0;
      for(i = 0; i < 6; i+=2) {
        // d1/d2 = gamma^2
        double d2 = 1.0;
        double d1 = gamma[iK]*gamma[iK]*d2;
        double d12 = d2+d1;
        //printf("gam: %4.3e; d1: %4.3e; d2: %4.3e\n", gamma[iK], d1, d2);
        d1 /= d12;
        //d2 /= d12;
        for(int d = 0; d < 3; d++) {
          double r12 = txp[i+1][d] - txp[i][d];
          xmp[d][iK] = txp[i][d]+r12*d1;
          xmp[d][iK] *= 10;
        }
        iK++;
      }
      // double det = MathExtra::det3(xmp);

      // calculate weight distribution to conserve CoM
      double rhoi[3];
      //LSCCoM(xmp,gX,igsum,rhoi);

      //MathExtra::mldivide3(xmp,gX,rhoi); // for A x = b (inputs: A, b, x)

      printf("rhoi: %4.3e, %4.3e, %4.3e\n", rhoi[0], rhoi[1], rhoi[2]);

      error->one(FLERR,"check");

    // shuffle particle list
    } else {
      for(i = npCluster-1; i > 0; i--) {
        j = random->uniform()*(i+1);
        std::swap(currentCluster[i], currentCluster[j]);
      }
    }

    // pick pairs and merge
    int ipid = 0;
    ip = 0;
    for(int iK = 0; iK < nK; iK++){
      // if eigenpair is zero, then their weights are zero
      //if(signK[iK] == 0) continue; // skip over zero eigenvalues

      // pick pairs
      ipart = &particles[currentCluster[ipid++]];
      jpart = &particles[currentCluster[ipid++]];

      ipart->sw = gi[ip];
      jpart->sw = gj[ip];

      // set new velocities and weights (double checked)
      for(int d = 0; d < 3; d++) {
        ipart->v[d] = vi[ip][d];
        jpart->v[d] = vj[ip][d];
        //printf("d: %i; ipart->v: %4.8e, jpart->v: %4.8e\n", d, ipart->v[d], jpart->v[d]);

        if(rFlag == 2) {
          ipart->x[d] = xm[d];
          jpart->x[d] = xm[d];
        }
      }
      ip++;
    }

    for(int idel = 0; idel < npCluster; idel++) {
      ipart = &particles[currentCluster[idel]];
      if(idel < 2*nK) continue;
      if (ndelete == maxdelete) {
        maxdelete += DELTADELETE;
        memory->grow(dellist,maxdelete,"collide:dellist");
      }
      ipart->sw = -1;
      dellist[ndelete++] = currentCluster[idel];
      dnRed++;
    }

  // Goskonov 2022
  } else if (reduction_type == 3) {
    if(npCluster <= nGono) return;
    int npremain = gonoskovMerge();

    //if(npremain != nGono) printf("not equal\n");
    //else printf("equal\n");

    // delete other particles // check correct number deleted
    for(int idel = npremain; idel < npCluster; idel++) {
      if (ndelete == maxdelete) {
        maxdelete += DELTADELETE;
        memory->grow(dellist,maxdelete,"collide:dellist");
      }
      ipart = &particles[currentCluster[idel]];
      ipart->sw = -1;
      dellist[ndelete++] = currentCluster[idel];
      dnRed++;
    }

  } else {
    error->one(FLERR,"no valid reduction method specified");
  }
  return;
}

/* ----------------------------------------------------------------------
    Redistribute weights using Gonoskov's algorithm to conserve
    certain moments. The mass(1), momentum(3), center of mass(3),
    and energy(1) are always conserved

    Returns the actual number of particles remaining.
------------------------------------------------------------------------- */
int Collide::gonoskovMerge()
{
  if(npCluster <= nGono) return npCluster;
  double *ixp, *ivp, igp;
  double *jxp, *jvp, jgp;
  Particle::OnePart *ipart;
  Particle::OnePart *jpart;
  Particle::OnePart *particles = particle->particles;
  // number of remaining particles
  // intialize with number in cluster
  int Nr = npCluster;

  // Moments (e)
  double e[Nr][nm], emag[nm];

  // find raw moments (rewrite later)
  // implicitly conserves central moments as well
  for(int i = 0; i < nm; i++) emag[i] = 0.0;
  for(int ip = 0; ip < Nr; ip++) {
    int im = 0;
    ipart = &particles[currentCluster[ip]];
    ixp = ipart->x;
    ivp = ipart->v;
    igp = ipart->sw;

    if(ipart->sw==0.0) error->one(FLERR,"zero weight particle found");

    // mass
    e[ip][im++] = 1.0;

    // CoM
    e[ip][im++] = ixp[0];
    e[ip][im++] = ixp[1];
    e[ip][im++] = ixp[2];

    // Momenta
    e[ip][im++] = ivp[0];
    e[ip][im++] = ivp[1];
    e[ip][im++] = ivp[2];

    // Energy
    e[ip][im++] = ivp[0]*ivp[0];
    e[ip][im++] = ivp[1]*ivp[1];
    e[ip][im++] = ivp[2]*ivp[2];

    // second raw spatial moment
    if(x2flag > 0) {
      e[ip][im++] = ixp[0]*ixp[0];
      e[ip][im++] = ixp[1]*ixp[1];
      e[ip][im++] = ixp[2]*ixp[2];
    }
      
    // off-diagonal stress tensor
    if(v2flag > 0) {
      e[ip][im++] = ivp[0]*ivp[1];
      e[ip][im++] = ivp[1]*ivp[2];
      e[ip][im++] = ivp[2]*ivp[0];
    }

    // heat flux
    if(v3flag > 0) {
      e[ip][im++] = ivp[0]*ivp[0]*ivp[0];
      e[ip][im++] = ivp[1]*ivp[1]*ivp[1];
      e[ip][im++] = ivp[2]*ivp[2]*ivp[2];
    }

    // find mag
    for(int i = 0; i < nm; i++) emag[i] += (e[ip][i]*e[ip][i]);
  }

  // normalize moment vectors to reduce numerical error
  for(int d = 0; d < nm; d++) {
    emag[d] = sqrt(emag[d]);
    for(int ip = 0; ip < Nr; ip++) {
      e[ip][d] /= emag[d];
    }
  }
  // reduce number of particles (or down-sample) until Nr = npC
  while(Nr > nGono) {
    // initialize balance vectors
    double vb[Nr][Nr]; // balance vectors
    for(int i = 0; i < Nr; i++) {
      for(int j = 0; j < Nr; j++) {
        if(i==j) vb[i][j] = 1.0;
        else vb[i][j] = 0.0;
      }
    }

    // update v (balance vectors)
    double dotv1,dotv2;
    double vnew[Nr];
    for(int k = 0; k < nm; k++) {
      for(int p = 0; p < Nr-k-1; p++) { // particle "i"
        dotv1 = dotv2 = 0.0;
        for(int ip = 0; ip < Nr; ip++) {
          dotv1 += vb[ip][p]*e[ip][k];
          dotv2 += vb[ip][p+1]*e[ip][k];
        }

        if(dotv2 == 0.0 && dotv1 == 0.0) continue;

        for(int ip = 0; ip < Nr; ip++) vnew[ip] = dotv2*vb[ip][p] - dotv1*vb[ip][p+1];
        double nvnew = 0.0;
        for(int ip = 0; ip < Nr; ip++) nvnew += (vnew[ip]*vnew[ip]);
        nvnew = sqrt(nvnew);
        if(nvnew > 0) {
          for(int ip = 0; ip < Nr; ip++) vnew[ip] /= nvnew;
          for(int ip = 0; ip < Nr; ip++) vb[ip][p] = vnew[ip];
        } // else use old values
      }
    }

    double vbmag = 0.0;
    for(int ip = 0; ip < Nr; ip++) vbmag += (vb[ip][0]*vb[ip][0]);
    if(vbmag == 0) return Nr;

    // first column is the balanced vector
    // find ap / am
    double am = 0.0;
    double ap = 0.0;
    for(int ip = 0; ip < Nr; ip++) { // rows
      ipart = &particles[currentCluster[ip]];
      double at = -ipart->sw/vb[ip][0];
      if(vb[ip][0] > 0) {
        if(am == 0.0) am = at;
        else if(at>am) am = at;
      } else if(vb[ip][0] < 0) {
        if(ap == 0.0) ap = at;
        else if(at<ap) ap = at;
      }
    }

    double pp = am/(am-ap);
    // zero out one particle weight and redistribute
    int pFlag;
    if(pp > random->uniform()) pFlag = 1;
    else pFlag = -1;

    for(int ip = 0; ip < Nr; ip++) {
      ipart = &particles[currentCluster[ip]];
      if(pFlag > 0) ipart->sw += ap*vb[ip][0];
      else ipart->sw += am*vb[ip][0];
    }

    //std::swap(currentCluster[imin], currentCluster[Nr-1]);
    int inext = 0;
    int tmp;
    for(int ip = 0; ip < Nr; ip++) {
      ipart = &particles[currentCluster[ip]];

      // move non-zero weights to front and zero weights to back
      if(ipart->sw > 1) {
        std::swap(currentCluster[ip],currentCluster[inext]);
        // swap the constraints to avoid remaking
        for(int im = 0; im < nm; im++) std::swap(e[ip][im], e[inext][im]);
        inext++;
      }
    }

    if(inext == Nr) {
      printf("none removed\n");
      return Nr;
    } else if(inext==nGono) {
      return inext;
    } else {
       Nr = inext;
    }
  }
  return Nr;
}


/* ----------------------------------------------------------------------
    Determine geometric mean using Weiszfield's algorithm
------------------------------------------------------------------------- */

// May need to rewrite this to compute distances relative to bottom of cell
// May compute a zero distance which would lead to a NaN
double Collide::weiszfeld(double &xp, double &yp, double &zp, int maxk, double tol)
{
  double *ixp, *ivp, igp;
  Particle::OnePart *ipart;
  Particle::OnePart *particles = particle->particles;

  double xtol = tol+1.0;
  int k = 0;

  double sumW, sumWX[3];
  double xold[3], xnew[3];
  double dx,dy,dz,dr;
  double errold,errnew;

  // guess
  xold[0] = xp;
  xold[1] = yp;
  xold[2] = zp;

  // Iterate until max number of iterations (Niter) or until the percent 
  // change in distance between two iterations is larger than the 
  // tolerance (tol)
  errold = 0.0;
  while(k < maxk && xtol > tol) {
    // Update iteration number
    k++;
    sumW = 0.0;
    sumWX[0] = sumWX[1] = sumWX[2] = 0.0;
    for(int ip = 0; ip < npCluster; ip++) {
      ipart = &particles[currentCluster[ip]];
      ixp = ipart->x;
      igp = ipart->sw;

      dx = ixp[0]-xold[0];
      dy = ixp[1]-xold[1];
      dz = ixp[2]-xold[2];
      dr = sqrt(dx*dx + dy*dy + dz*dz);

      sumW += (igp/dr);
      for(int d = 0; d < 3; d++) sumWX[d] += (igp*ixp[d]/dr);
    }

    for(int d = 0; d < 3; d++) xnew[d] = sumWX[d]/sumW;

    // if error changes by less than tolerance, converged
    errnew = 0.0;
    for(int ip = 0; ip < npCluster; ip++) {
      ipart = &particles[currentCluster[ip]];
      ixp = ipart->x;
      igp = ipart->sw;
      errnew += igp*dist(xp,yp,zp,ixp[0],ixp[1],ixp[2]);
    }
    xtol = abs(errnew-errold)/errnew;
    errold = errnew;

    // Set new guess as old guess
    for(int d = 0; d < 3; d++) xold[d] = xnew[d];
  }

  // Return approximated geometric mean
  xp = xnew[0];
  yp = xnew[1];
  zp = xnew[2];
  return errnew;
}
