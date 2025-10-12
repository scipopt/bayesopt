/*
-------------------------------------------------------------------------
   This file is part of BayesOpt, an efficient C++ library for 
   Bayesian optimization.

   Copyright (C) 2011-2015 Ruben Martinez-Cantin <rmcantin@unizar.es>
 
   BayesOpt is free software: you can redistribute it and/or modify it 
   under the terms of the GNU Affero General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   BayesOpt is distributed in the hope that it will be useful, but 
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Affero General Public License for more details.

   You should have received a copy of the GNU Affero General Public License
   along with BayesOpt.  If not, see <http://www.gnu.org/licenses/>.
------------------------------------------------------------------------
*/

#include "log.hpp"
#include "posteriormodel.hpp"
#include "posterior_fixed.hpp"
#include "posterior_empirical.hpp"
#include "posterior_mcmc.hpp"

namespace bayesopt
{

  PosteriorModel* PosteriorModel::create(size_t dim, Parameters& params, randEngine& eng)
  {
    switch (params.l_type)
      {
      case L_FIXED: return new PosteriorFixed(dim,params,eng);
      case L_EMPIRICAL: return new EmpiricalBayes(dim,params,eng);
      case L_DISCRETE: // TODO:return new
      case L_MCMC: return new MCMCModel(dim,params,eng);
      case L_ERROR:
      default:
	throw std::invalid_argument("Learning type not supported");
      }
  };

  PosteriorModel::PosteriorModel(size_t dim, Parameters& parameters,
				 randEngine& eng):
    mParameters(parameters), mDims(dim), mMean(dim, parameters)
  {} 

  PosteriorModel::~PosteriorModel()
  { } // Default destructor


  void PosteriorModel::setSamples(const matrixd &x, const vectord &y)
  { 
    mData.setSamples(x,y);  
    mMean.setPoints(mData.mX);  //Because it expects a vecOfvec instead of a matrixd 
  }

  void PosteriorModel::setSamples(const matrixd &x)
  { 
    mData.setSamples(x);  
    mMean.setPoints(mData.mX);  //Because it expects a vecOfvec instead of a matrixd 
  }

  void PosteriorModel::setSamples(const vectord &y)
  { 
    mData.setSamples(y);  
  }

  void PosteriorModel::setSamples(const vecOfvec& x, const vectord& y)
  {
    matrixd xx(x.size(), x.empty() ? 0 : x[0].size());

    for( size_t i = 0; i < x.size(); ++i )
      row(xx, i) = x[i];

    mData.setSamples(xx, y);
    mMean.setPoints(mData.mX);
  }

  void PosteriorModel::setSample(const vectord &x, double y)
  { 
    matrixd xx(1,x.size());  vectord yy(1);
    row(xx,0) = x;           yy(0) = y;
    mData.setSamples(xx,yy);   
    mMean.setPoints(mData.mX);  //Because it expects a vecOfvec instead of a matrixd
  }

  void PosteriorModel::addSample(const vectord &x, double y)
  {  mData.addSample(x,y); mMean.addNewPoint(x);  };

  void PosteriorModel::storeEvaluationMeans()
  {
    mData.values = vecOfvec(mData.mY.size());
    mData.indices.resize(mData.mX.size());
    std::iota(mData.indices.begin(), mData.indices.end(), 0);
    std::sort(mData.indices.begin(), mData.indices.end(), [this](size_t i, size_t j)
    {
      if( std::lexicographical_compare(this->mData.mX[i].begin(), this->mData.mX[i].end(), this->mData.mX[j].begin(), this->mData.mX[j].end()) )
        return true;
      if( std::lexicographical_compare(this->mData.mX[j].begin(), this->mData.mX[j].end(), this->mData.mX[i].begin(), this->mData.mX[i].end()) )
        return false;
      return i < j;
    });

    size_t beginposition = 0;
    size_t endposition = 1;

    if( !mData.indices.empty() )
    {
      mData.values[mData.indices[beginposition]].resize(mData.values[mData.indices[beginposition]].size() + 1);
      mData.values[mData.indices[beginposition]][mData.values[mData.indices[beginposition]].size() - 1] = mData.mY[mData.indices[beginposition]];
    }

    while( endposition < mData.indices.size() )
    {
      if( !std::equal(mData.mX[mData.indices[beginposition]].begin(), mData.mX[mData.indices[beginposition]].end(), mData.mX[mData.indices[endposition]].begin()) )
      {
        ++beginposition;
        mData.indices[beginposition] = mData.indices[endposition];
      }

      mData.values[mData.indices[beginposition]].resize(mData.values[mData.indices[beginposition]].size() + 1);
      mData.values[mData.indices[beginposition]][mData.values[mData.indices[beginposition]].size() - 1] = mData.mY[mData.indices[endposition]];
      ++endposition;
    }

    mData.indices.resize(beginposition + 1);

    for( const auto& index : mData.indices )
    {
      double sum = 0.0;

      for( size_t i = 0; i < mData.values[index].size(); ++i )
      {
        sum += mData.values[index][i];
        mData.values[index][i] = sum / (i + 1);
      }
    }
  }

  void PosteriorModel::updateMinMax()
  {
    storeEvaluationMeans();

    double minmean = std::numeric_limits<double>::infinity();
    double maxmean = -std::numeric_limits<double>::infinity();
    size_t minnumb = 0;
    size_t maxnumb = 0;

    mData.mMinIndex = 0;
    mData.mMaxIndex = 0;

    for( const auto& index : mData.indices )
    {
      if( minmean > mData.values[index][mData.values[index].size() - 1] || ( minmean == mData.values[index][mData.values[index].size() - 1] && ( minnumb < mData.values[index].size() || ( minnumb == mData.values[index].size() && mData.mMinIndex < index ) ) ) )
      {
        minmean = mData.values[index][mData.values[index].size() - 1];
        minnumb = mData.values[index].size();
        mData.mMinIndex = index;
      }

      if( maxmean < mData.values[index][mData.values[index].size() - 1] || ( maxmean == mData.values[index][mData.values[index].size() - 1] && ( maxnumb < mData.values[index].size() || ( maxnumb == mData.values[index].size() && mData.mMaxIndex < index ) ) ) )
      {
        maxmean = mData.values[index][mData.values[index].size() - 1];
        maxnumb = mData.values[index].size();
        mData.mMaxIndex = index;
      }
    }
  }

  vecOfvec PosteriorModel::getPointsAtMinimum()
  {
    storeEvaluationMeans();

    std::sort(mData.indices.begin(), mData.indices.end(), [this](size_t i, size_t j)
    {
      if( this->mData.values[i][this->mData.values[i].size() - 1] != this->mData.values[j][this->mData.values[j].size() - 1] )
        return this->mData.values[i][this->mData.values[i].size() - 1] < this->mData.values[j][this->mData.values[j].size() - 1];
      if( this->mData.values[i].size() != this->mData.values[j].size() )
        return this->mData.values[i].size() > this->mData.values[j].size();
      return i > j;
    });
    assert(mData.indices.empty() || mData.indices[0] == mData.mMinIndex);

    vecOfvec results;

    results.reserve(mData.indices.size());

    for( const auto& index : mData.indices )
      results.push_back(mData.mX[index]);

    return results;
  }
} //namespace bayesopt

