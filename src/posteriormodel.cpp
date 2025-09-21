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

  void PosteriorModel::getEvaluationMeans(vectord& values, vectori& nvalues)
  {
    values = vectord(mData.mY.size(), 0.0);
    nvalues = vectori(mData.mY.size(), 0);
    mData.indices.resize(mData.mX.size());
    std::iota(mData.indices.begin(), mData.indices.end(), 0);
    std::sort(mData.indices.begin(), mData.indices.end(), [this](size_t i, size_t j)
    {
      if( std::lexicographical_compare(this->mData.mX[i].begin(), this->mData.mX[i].end(), this->mData.mX[j].begin(), this->mData.mX[j].end()) )
        return true;
      if( std::lexicographical_compare(this->mData.mX[j].begin(), this->mData.mX[j].end(), this->mData.mX[i].begin(), this->mData.mX[i].end()) )
        return false;
      if( this->mData.mY[i] != this->mData.mY[j] )
        return this->mData.mY[i] < this->mData.mY[j];
      return i > j;
    });

    size_t beginposition = 0;
    size_t endposition = 1;

    if( !mData.indices.empty() )
    {
      values[mData.indices[beginposition]] += mData.mY[mData.indices[beginposition]];
      ++nvalues[mData.indices[beginposition]];
    }

    while( endposition < mData.indices.size() )
    {
      if( !std::equal(mData.mX[mData.indices[beginposition]].begin(), mData.mX[mData.indices[beginposition]].end(), mData.mX[mData.indices[endposition]].begin()) )
      {
        ++beginposition;
        mData.indices[beginposition] = mData.indices[endposition];
      }

      values[mData.indices[beginposition]] += mData.mY[mData.indices[endposition]];
      ++nvalues[mData.indices[beginposition]];
      ++endposition;
    }

    mData.indices.resize(beginposition + 1);

    for( const auto& index : mData.indices )
      values[index] /= nvalues[index];
  }

  void PosteriorModel::updateMinMax()
  {
    vectord values;
    vectori nvalues;

    getEvaluationMeans(values, nvalues);

    double minmean = std::numeric_limits<double>::infinity();
    double maxmean = -std::numeric_limits<double>::infinity();
    size_t minnumb = 0;
    size_t maxnumb = 0;

    mData.mMinIndex = 0;
    mData.mMaxIndex = 0;

    for( const auto& index : mData.indices )
    {
      if( minmean > values[index] || ( minmean == values[index] && ( minnumb < nvalues[index] || ( minnumb == nvalues[index] && mData.mMinIndex < index ) ) ) )
      {
        minmean = values[index];
        minnumb = nvalues[index];
        mData.mMinIndex = index;
      }

      if( maxmean < values[index] || ( maxmean == values[index] && ( maxnumb < nvalues[index] || ( maxnumb == nvalues[index] && mData.mMaxIndex < index ) ) ) )
      {
        maxmean = values[index];
        maxnumb = nvalues[index];
        mData.mMaxIndex = index;
      }
    }
  }

  vecOfvec PosteriorModel::getPointsAtMinimum()
  {
    vectord values;
    vectori nvalues;

    getEvaluationMeans(values, nvalues);

    std::sort(mData.indices.begin(), mData.indices.end(), [&values, &nvalues](size_t i, size_t j)
    {
      if( values[i] != values[j] )
        return values[i] < values[j];
      if( nvalues[i] != nvalues[j] )
        return nvalues[i] > nvalues[j];
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

