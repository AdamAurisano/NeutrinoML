
#include "CAFAna/Experiment/LikelihoodCovMxExperiment.h"

#include "CAFAna/Core/EigenUtils.h"
#include "CAFAna/Core/LoadFromFile.h"

#include "OscLib/IOscCalc.h"
#include "OscLib/OscCalcSterile.h" // for debugging

#include "TDirectory.h"
#include "TObjString.h"
#include "TH1.h"

#include <iomanip>
#include <iostream>
#include <chrono>
#include <cmath> // for isnan

#include <Eigen/Dense>

namespace ana
{
  // Standard C++ namespace
  using std::count;
  using std::cout;
  using std::endl;
  using std::fixed;
  using std::get;
  using std::isnan;
  using std::runtime_error;
  using std::setprecision;
  using std::string;
  using std::vector;

  // Standard C++ chrono namespace
  using std::chrono::high_resolution_clock;
  using std::chrono::duration_cast;
  using std::chrono::milliseconds;
  using std::chrono::seconds;
  using std::chrono::minutes;

  // Eigen tools
  using Eigen::ArrayXd;
  using Eigen::MatrixXd;
  using Eigen::VectorXd;

  // CAFAna covariance matrix namespace
  using covmx::Component;
  using covmx::Sample;
  using covmx::CovarianceMatrix;

  //----------------------------------------------------------------------
  LikelihoodCovMxExperiment::LikelihoodCovMxExperiment(vector<Sample> samples,
    CovarianceMatrix* covmx, double epsilon, double lambdazero, double nu)
  : fSamples(samples), fCovMx(covmx), fNBins(0), fNBinsFull(0), fLambdaZero(lambdazero),
    fNu(nu), fBetaHist(nullptr), fMuHist(nullptr),
    fBetaMuHist(nullptr), fPredShiftedHist(nullptr), fPredUnshiftedHist(nullptr),
    fDataHist(nullptr), fResetShifts(true), fVerbose(false)
  {
    for (Sample s: fSamples) {
      fNBins += s.GetBinning().NBins();
      fNBinsFull += s.GetBinning().NBins() * GetComponents(s).size();
      fNBinsPerSample.push_back(s.GetBinning().NBins());
      fComps.push_back(GetComponents(s));
    }

    fGrad.resize(fNBinsFull);
    fHess.resize(fNBinsFull, fNBinsFull);

    // Set up beta and mu
    fMu.resize(fNBinsFull);
    fBeta.resize(fNBinsFull);
    fBetaMask.resize(fNBinsFull);
    for (size_t i = 0; i < fNBinsFull; ++i) {
      fBeta(i) = 1.;
    }

    // Fill the data vector
    fData = ArrayXd::Zero(fNBins);
    fCosmic = ArrayXd::Zero(fNBins);
    size_t i = 0;
    for (size_t iS = 0; iS < fSamples.size(); ++iS) {
      double pot      = fSamples[iS].GetData()->POT();
      double livetime = fSamples[iS].GetData()->Livetime();
      TH1D* hData     = fSamples[iS].GetData()->ToTH1(pot);
      TH1D* hCos      = fSamples[iS].HasCosmic() ?
        fSamples[iS].GetCosmic()->ToTH1(livetime, kLivetime) : nullptr;
      for (size_t iBin = 1; iBin <= fNBinsPerSample[iS]; ++iBin) {
        fData(i) = hData->GetBinContent(iBin);
        fCosmic(i) = hCos ? hCos->GetBinContent(iBin) : 0;
        ++i;
      } // for bin i
      delete hData;
      delete hCos;
    } // for sample iS

    if (fCovMx) {
      // Invert matrix ahead of time
      MatrixXd mx = CovarianceMatrix::ForcePosDef(fCovMx->GetFullCovMx(), epsilon);
      fMxInv = mx.inverse();
      MatrixXd id = mx;
      id.setIdentity();
      fResidual = ((mx.transpose() * fMxInv) - id).maxCoeff();
      if (fVerbose) cout << "Eigen matrix inversion residual is " << fResidual << endl;
    }

  }

  //----------------------------------------------------------------------
  double LikelihoodCovMxExperiment::ChiSq(osc::IOscCalcAdjustable* osc,
    const SystShifts& syst) const
  {
    DontAddDirectory guard;

    // Get MC and data spectra
    size_t i = 0;
    for (size_t iS = 0; iS < fSamples.size(); ++iS) {
      double pot = fSamples[iS].GetData()->POT();
      SystShifts shift = fSamples[i].GetSystShifts(syst);
      for (Component comp: fComps[iS]) {
        Spectrum spec = fSamples[iS].GetPrediction()->PredictComponentSyst(osc,
    shift, get<0>(comp), get<1>(comp), get<2>(comp));
        TH1D* hMu = spec.ToTH1(pot);
        for (size_t iBin = 1; iBin <= (size_t)hMu->GetNbinsX(); ++iBin) {
    fMu(i) = hMu->GetBinContent(iBin);
    ++i;
        } // for bin
        delete hMu;
      } // for component
    } // for sample

    // Catch errors in prediction, to disambiguate from fitting problems
    osc::OscCalcSterile* calc = dynamic_cast<osc::OscCalcSterile*>(osc);
    if (calc && fMu.array().isNaN().any()) { 
      cout << "ERROR! NAN VALUES IN PREDICTION!" << endl;
      cout << "---------- OSCILLATION PARAMETERS ----------" << endl
           << "Dm21:     " << calc->GetDm(2)       << endl
           << "Dm31:     " << calc->GetDm(3)       << endl
           << "Dm41:     " << calc->GetDm(4)       << endl
           << "Theta 13: " << calc->GetAngle(1, 3) << endl
           << "Theta 23: " << calc->GetAngle(2, 3) << endl
           << "Theta 14: " << calc->GetAngle(1, 4) << endl
           << "Theta 24: " << calc->GetAngle(2, 4) << endl
           << "Theta 34: " << calc->GetAngle(3, 4) << endl
           << "Delta 13: " << calc->GetDelta(1, 3) << endl
           << "Delta 24: " << calc->GetDelta(2, 4) << endl
           << "--------------------------------------------" << endl;
    }

    if (fResetShifts) {
      for (size_t i = 0; i < fNBinsFull; ++i) fBeta(i) = 1.;
    }

    // Do stats-only calculation if no matrix
    if (!fCovMx) {
      ArrayXd e = GetExpectedSpectrum();
      return ana::LogLikelihood(e, fData);
    }

    // Number of bins
    if (fCovMx && fNBins != (size_t)fCovMx->GetBinning().NBins()) {
      throw runtime_error(
        "Number of bins in predictions does not match covariance matrix!");
    }

    // Fill histogram before beta shifting
    if (fPredUnshiftedHist) {
      ArrayXd e = GetExpectedSpectrum();
      for (size_t j = 0; j < fNBins; ++j) {
        fPredUnshiftedHist->SetBinContent(j+1, e(j));
      }
    } // if filling unshifted spectrum

    return LikelihoodCovMxNewton();

  } // function LikelihoodCovMxExperiment::ChiSq

  void LikelihoodCovMxExperiment::Reset() const {
    cout << "Resetting syst shifts due to new seed." << endl;
    fBeta = ArrayXd::Constant(fNBinsFull, 1);
  }

  //---------------------------------------------------------------------------
  ArrayXd LikelihoodCovMxExperiment::GetExpectedSpectrum() const {

    ArrayXd ret = ArrayXd::Zero(fNBins);
    unsigned int fullOffset(0), scaledOffset(0);

    for (size_t iS = 0; iS < fSamples.size(); ++iS) { // loop over samples
      for (size_t iC = 0; iC < fComps[iS].size(); ++iC) { // loop over components
        for (size_t iB = 0; iB < fNBinsPerSample[iS]; ++iB) { // loop over bins
    unsigned int iScaled = scaledOffset + iB;
    unsigned int iFull = fullOffset + (iC*fNBinsPerSample[iS]) + iB;

    ret(iScaled) += fMu(iFull)*fBeta(iFull);
        } // for bin
      } // for component
      // Now loop one more time to add cosmics
      for (size_t iB = 0; iB < fNBinsPerSample[iS]; ++iB) {
        unsigned int iScaled = scaledOffset + iB;
        ret(iScaled) += fCosmic(iScaled);
      } // for bin
      scaledOffset += fNBinsPerSample[iS]; // increase offset in scaled matrix
      fullOffset += fComps[iS].size() * fNBinsPerSample[iS];
    } // for sample
    return ret;

  } // function LikelihoodCovMxExperiment::GetExpectedSpectrum

  //---------------------------------------------------------------------------
  double LikelihoodCovMxExperiment::GetChiSq(ArrayXd e) const {

    // the statistical likelihood of the shifted spectrum...
    fStatChiSq = ana::LogLikelihood(e, fData);

    // ...plus the penalty the prediction picks up from its covariance
    VectorXd vec = fBeta - 1.;
    fSystChiSq = vec.transpose() * fMxInv * vec;
    return fStatChiSq + fSystChiSq;

  } // function LikelihoodCovMxExperiment::GetChiSq

  //---------------------------------------------------------------------------
  void LikelihoodCovMxExperiment::InitialiseBetas() const{
    double eps(1e-40);
    unsigned int fullOffset(0), scaledOffset(0);
    for (size_t iS = 0; iS < fSamples.size(); ++iS) { // loop over samples
      for (size_t iB = 0; iB < fNBinsPerSample[iS]; ++iB) { // loop over bins
        unsigned int iScaled = scaledOffset+iB;
        double sum = fCosmic(iScaled);
        for (size_t iC = 0; iC < fComps[iS].size(); ++iC) { // loop over components
    unsigned int iFull = fullOffset+(iC*fNBinsPerSample[iS])+iB;
    sum += fMu(iFull);
        } // for component
        double val = fData(iScaled) / (sum + eps);
        for (size_t iC = 0; iC < fComps[iS].size(); ++iC) { // loop over components
    unsigned int iFull = fullOffset+(iC*fNBinsPerSample[iS])+iB;
    fBeta(iFull) = val;
        } // for component
      } // for bin
      scaledOffset += fNBinsPerSample[iS]; // increase offset in scaled matrix
      fullOffset += fComps[iS].size() * fNBinsPerSample[iS];
    } // for sample

  } // function LikelihoodCovMxExperiment::InitialiseBetas

  //---------------------------------------------------------------------------
  bool LikelihoodCovMxExperiment::MaskBetas() const {

    bool maskChange = false;

    size_t iFullOffset(0.), iScaledOffset(0.);
    for (size_t iS = 0; iS < fSamples.size(); ++iS) { // loop over samples
      for (size_t iC = 0; iC < fComps[iS].size(); ++iC) { // loop over components
        for (size_t iB = 0; iB < fNBinsPerSample[iS]; ++iB) { // loop over bins
    // Get the current scaled & full bin
    unsigned int iScaled = iScaledOffset+iB;
    unsigned int iFull = iFullOffset+(iC*fNBinsPerSample[iS])+iB;

    double y = fCosmic(iScaled);
    for (size_t jC = 0; jC < fComps[iS].size(); ++jC) {
      if (iC != jC) { // beta mu from this bin is 0
        unsigned int jFull = iFullOffset+(jC*fNBinsPerSample[iS])+iB;
        y += fMu(jFull) * fBeta(jFull);
      }
    } // for component j
    if (y < 1e-40) y = 1e-40; // clamp y

    double z(0.); // calculate z
    for (size_t jFull = 0; jFull < fNBinsFull; ++jFull) {
      double beta = (jFull == iFull) ? 0 : fBeta[jFull]; // set this beta to 0
      z += (beta-1)*fMxInv(iFull, jFull);
    }

    // Calculate gradient
    double gradZero = 2 * (fMu(iFull)*(1.0-(fData(iScaled)/y)) + z);
    if (gradZero > 0 && fBetaMask[iFull]) { // if we're masking a bin off
      fBetaMask[iFull] = false;
      maskChange = true;
    }
    if (gradZero <= 0 && !fBetaMask[iFull]) { // if we're masking a bin on
      fBetaMask[iFull] = true;
      fBeta(iFull) = 0;
      maskChange = true;
    }
        } // for bin i
      } // for component i
      iScaledOffset += fNBinsPerSample[iS]; // increase offset in scaled matrix
      iFullOffset += fComps[iS].size() * fNBinsPerSample[iS];
    } // for sample i

    return maskChange;

  } // function LikelihoodCovMxExperiment::MaskBetas

  //---------------------------------------------------------------------------
  void LikelihoodCovMxExperiment::GetGradAndHess() const {

    size_t iFullOffset(0.), iScaledOffset(0.);
    for (size_t iS = 0; iS < fSamples.size(); ++iS) { // loop over samples
      for (size_t iC = 0; iC < fComps[iS].size(); ++iC) { // loop over components
        for (size_t iB = 0; iB < fNBinsPerSample[iS]; ++iB) { // loop over bins
    // Get the current scaled & full bin
    unsigned int iScaled = iScaledOffset+iB;
    unsigned int iFull = iFullOffset+(iC*fNBinsPerSample[iS])+iB;
    double y = fCosmic(iScaled); // calculate y
    for (size_t jC = 0; jC < fComps[iS].size(); ++jC) {
      unsigned int jFull = iFullOffset+(jC*fNBinsPerSample[iS])+iB;
      y += fMu(jFull) * fBeta(jFull);
    } // for component j
    if (y < 1e-40) y = 1e-40; // clamp y
    double z(0.); // calculate z
    for (size_t jFull = 0; jFull < fNBinsFull; ++jFull) {
      z += (fBeta(jFull)-1)*fMxInv(iFull, jFull);
    }

    // Calculate gradient
    fGrad(iFull) = 2 * (fMu(iFull)*(1.0-(fData(iScaled)/y)) + z);

    // Populate Hessian matrix
    size_t jFullOffset(0.), jScaledOffset(0.);
    for (size_t jS = 0; jS < fSamples.size(); ++jS) { // loop over samples
      for (size_t jC = 0; jC < fComps[jS].size(); ++jC) { // loop over components
        for (size_t jB = 0; jB < fNBinsPerSample[jS]; ++jB) { // loop over bins
    // Get the current scaled & full bin
    unsigned int jScaled = jScaledOffset + jB;
    unsigned int jFull = jFullOffset+(jC*fNBinsPerSample[jS])+jB;
    if (iScaled != jScaled) {
      fHess(iFull, jFull) = 2 * fMxInv(iFull, jFull);
    } else {
      fHess(iFull, jFull) = 2 * (fMu(iFull)*fMu(jFull)*(fData(iScaled)/(y*y))
        + fMxInv(iFull, jFull));
    }
        } // for bin j
      } // for component j
      jScaledOffset += fNBinsPerSample[jS]; // increase offset in scaled matrix
      jFullOffset += fComps[jS].size() * fNBinsPerSample[jS];
    } // for sample j
        } // for bin i
      } // for component i
      iScaledOffset += fNBinsPerSample[iS]; // increase offset in scaled matrix
      iFullOffset += fComps[iS].size() * fNBinsPerSample[iS];
    } // for sample i

    // Apply transformation and Levenberg-Marquardt
    for (size_t i = 0; i < fNBinsFull; ++i) {
      if (fUseLMA) fHess(i, i) *= 1. + fLambda;
      fGrad(i) *= -1;
    }

  } // function LikelihoodCovMxExperiment::GetGradAndHess

  //---------------------------------------------------------------------------
  void LikelihoodCovMxExperiment::GetReducedGradAndHess() const {

    // Mask off bins
    int maskedSize = count(fBetaMask.begin(), fBetaMask.end(), true);

    GetGradAndHess();

    if (fGradReduced.rows() != maskedSize) {
      fGradReduced.resize(maskedSize);
      fHessReduced.resize(maskedSize, maskedSize);
    }

    size_t c = 0;
    for (size_t i = 0; i < fNBinsFull; ++i) {
      if (fBetaMask[i]) {
        fGradReduced(c) = fGrad(i);
        ++c;
      }
    }
    size_t ci = 0;
    for (size_t i = 0; i < fNBinsFull; ++i) {
      if (fBetaMask[i]) {
        size_t cj = 0;
        for (size_t j = 0; j < fNBinsFull; ++j) {
    if (fBetaMask[j]) {
      fHessReduced(ci, cj) = fHess(i, j);
      ++cj;
    } // if j unmasked
        } // for j
        ++ci;
      } // if i unmasked
    } // for i
    
  } // function LikelihoodCovMxExperiment::GetReducedGradAndHess

  //---------------------------------------------------------------------------
  double LikelihoodCovMxExperiment::LikelihoodCovMxNewton() const {

    auto start = high_resolution_clock::now();

    // Start with all systematic shifts unmasked
    for (size_t i = 0; i < fNBinsFull; ++i) fBetaMask[i] = true;

    // We're trying to solve for the best expectation in each bin and each component. A good seed
    // value is 1 (no shift).
    // Note: beta will contain the best fit systematic shifts at the end of the process.
    // This should be saved to show true post-fit agreement
    const int maxIters = 5e3;
    fIteration = 0;
    int nFailures = 0;
    // InitialiseBetas();
    MaskBetas();
    fUseLMA = false;
    ResetLambda(); // Initialise lambda at starting value
    bool failed = false;

    // Keep a history of masks and chi squares to help avoid infinite loops
    bool keepChangingMask = true;
    vector<vector<bool>> maskHistory;
    vector<double> chisqHistory;

    ArrayXd e = GetExpectedSpectrum();
    double prev = GetChiSq(e);
    double minChi = prev;
    if (fVerbose) {
      cout << "Before minimization, chisq is " << prev
           << " (" << fStatChiSq << " stat, " << fSystChiSq << " syst)" << endl;
    }

    while (true) {
      ++fIteration;
      if (!fUseLMA && fIteration > 10) EnableLMA();
      if (fIteration > maxIters) {
        cout << "No convergence after " << maxIters << " iterations! Quitting out of the infinite loop." << endl;
        return minChi;
      }

      // If we have negative betas, mask them out
      GetReducedGradAndHess();
      VectorXd deltaBeta = fHessReduced.ldlt().solve(fGradReduced);
      if (deltaBeta.array().isNaN().any()) {
        int nanCounter = 0;
        for (int i = 0; i < deltaBeta.size(); ++i) {
          if (isnan(deltaBeta[i])) ++nanCounter;
        }
        if (fUseLMA) IncreaseLambda();
        EnableLMA();
        if (fVerbose) {
          cout << "LDLT solution failed! There are " << nanCounter << "nan delta betas. Resetting all betas.";
        }
        if (failed) InitialiseBetas();
        failed = true;
        continue;
      }

      // if (fVerbose) {
      //   double relativeError = (fHessReduced * deltaBeta - fGradReduced).norm() / fGradReduced.norm();
      //   cout << "Relative error of linear solution is " << relativeError << endl;
      // }

      size_t counter = 0;
      for (size_t i = 0; i < fNBinsFull; ++i) {
        if (fBetaMask[i]) {
          fBeta(i) += deltaBeta(counter);
          if (fBeta(i) < 0) { // if we pulled negative, mask this beta off
            fBeta(i) = 0;
            fBetaMask[i] = false;
          } else if (fBeta(i) > 1e10) { // if we pull too high, mask this beta off
            fBeta(i) = 1;
            fBetaMask[i] = false;
          } 
          ++counter;
        }
      } // for bin

      // Predict collapsed spectrum
      e = GetExpectedSpectrum();

      // Update the chisq
      double chisq = GetChiSq(e);

      if (isnan(chisq) || chisq > 1e20) {
        if (fVerbose && isnan(chisq))
          cout << "ChiSq is NaN! Resetting minimisation." << endl;
        else if (fVerbose && chisq > 1e20) {
          cout << "ChiSq is anomalously large! Resetting minimisation." << endl;
          vector<double> changes(deltaBeta.size());
          VectorXd::Map(&changes[0], deltaBeta.size()) = deltaBeta;
          std::sort(changes.begin(), changes.end(), std::greater<double>());
          cout << "  five greatest deltas:";
          for (int i = 0; i < 5; ++i) cout << "  " << changes[i];
          cout << endl;
        }
        ++nFailures;
        ResetLambda();
        fLambda *= (1 + ((double)nFailures/10.)); // push us out of fragile failure states
        EnableLMA();
        if (failed) InitialiseBetas();
        failed = true;
        continue;
      }

      if (chisq < minChi) minChi = chisq;

      // Bump down the LMA lambda
      if (fIteration > 1 && fUseLMA && chisq < prev) DecreaseLambda();

      if (fVerbose) {
        cout << "Iteration " << fIteration << ", chisq " << chisq
             << " (" << fStatChiSq << " stat, " << fSystChiSq << " syst)" << endl;
      }

      if (fSystChiSq < 0) {
        assert(false && "Negative systematic penalty!");
      }

      // If the updates didn't change anything at all then we're done
      double change = fabs(chisq-prev);

      // Converge to third decimal place
      if (change/chisq < 1e-3 || change < 1e-10) {

        bool maskChange = keepChangingMask ? MaskBetas() : false;

        if (maskChange) { // re-mask betas - if the mask changed, go again

          // this block of code prevents infinte loops thru bin masks
          maskHistory.push_back(fBetaMask);
          chisqHistory.push_back(chisq);
          for (size_t i = 0; i < maskHistory.size()-1; ++i) {
            if (std::equal(maskHistory[i].begin(), maskHistory[i].end(), fBetaMask.begin())) {
              if (fVerbose) cout << "Found a repeated mask! Setting the mask that provided the best chisq" << endl;
              double chi = 1e100;
              size_t pos = 999;
              for (size_t j = 0; j < chisqHistory.size(); ++j) {
                if (chisqHistory[j] < chi) {
                  chi = chisqHistory[j];
                  pos = j;
                }
              }
              fBetaMask = maskHistory[pos];
              for (size_t i = 0; i < fNBinsFull; ++i) if (!fBetaMask[i]) fBeta[i] = 0;
              keepChangingMask = false;
              break;
            }
          }

          ResetLambda();
          if (fVerbose) {
            cout << "Minimisation round finished with chisq " << chisq << ", bin mask:";
            for (size_t i = 0; i < fNBinsFull; ++i) if (!fBetaMask[i]) cout << " " << i+1;
            cout << endl;
          }
        }

        // Converge to tenth significant figure or tenth decimal place, whichever is larger
        // Alternatively, return smaller chisq if we're flip-flopping between two beta masks
        if (!maskChange && (change/chisq < 1e-10 || change < 1e-10)) {

          if (fVerbose) {
            cout << "Converged to " << chisq << " (" << fStatChiSq << " stat, " << fSystChiSq
              << " syst) after " << fIteration << " iterations and ";
            auto end = high_resolution_clock::now();
            double secs = duration_cast<seconds>(end-start).count();
            if (secs < 1) {
              double millisecs = duration_cast<milliseconds>(end-start).count();
              cout << millisecs << " ms." << endl;
            } else if (secs < 60) cout << secs << " seconds." << endl;
            else {
              double mins = duration_cast<minutes>(end-start).count();
              cout << mins << " minutes." << endl;
            }
          }

          // Fill histograms if necessary
          if (fBetaHist)        for (size_t i = 0; i < fNBinsFull; ++i) fBetaHist->SetBinContent(i+1, fBeta(i));
          if (fMuHist)          for (size_t i = 0; i < fNBinsFull; ++i) fMuHist->SetBinContent(i+1, fMu(i));
          if (fBetaMuHist)      for (size_t i = 0; i < fNBinsFull; ++i) fBetaMuHist->SetBinContent(i+1, fBeta(i)*fMu(i));
          if (fPredShiftedHist) for (size_t i = 0; i < fNBins; ++i) fPredShiftedHist->SetBinContent(i+1, e[i]);
  	
          return chisq;

        } else {
          if (fUseLMA) DecreaseLambda(); // If we're getting close to converging, speed things up
        } 
      }

      prev = chisq;

    } // end while
    
  } // function LikelihoodCovMxExperiment::LogLikelihoodCovMxNewton

  //----------------------------------------------------------------------
  void LikelihoodCovMxExperiment::SaveHists(bool opt) {
    if (!fCovMx) return;
    // Start by clearing up existing beta histogram
    if (fBetaHist) {
      delete fBetaHist;
      fBetaHist = nullptr;
    }
    if (fMuHist) {
      delete fMuHist;
      fMuHist = nullptr;
    }
    if (fBetaMuHist) {
      delete fBetaMuHist;
      fBetaMuHist = nullptr;
    }
    if (fPredShiftedHist) {
      delete fPredShiftedHist;
      fPredShiftedHist = nullptr;
    }
    if (fPredUnshiftedHist) {
      delete fPredUnshiftedHist;
      fPredUnshiftedHist = nullptr;
    }
    if (fDataHist) {
      delete fDataHist;
      fDataHist = nullptr;
    }
    // Now re-instantiate it if necessary
    if (opt) {
      Binning fullBins   = fCovMx->GetFullBinning();
      Binning scaledBins = fCovMx->GetBinning();
      fBetaHist          = new TH1D(UniqueName().c_str(), ";;", fullBins.NBins(), &fullBins.Edges()[0]);
      fMuHist            = new TH1D(UniqueName().c_str(), ";;", fullBins.NBins(), &fullBins.Edges()[0]);
      fBetaMuHist        = new TH1D(UniqueName().c_str(), ";;", fullBins.NBins(), &fullBins.Edges()[0]);
      fPredShiftedHist   = new TH1D(UniqueName().c_str(), ";;", scaledBins.NBins(), &scaledBins.Edges()[0]);
      fPredUnshiftedHist = new TH1D(UniqueName().c_str(), ";;", scaledBins.NBins(), &scaledBins.Edges()[0]);
      fDataHist          = new TH1D(UniqueName().c_str(), ";;", scaledBins.NBins(), &scaledBins.Edges()[0]);
      for (size_t i = 0; i < fNBins; ++i) {
        fDataHist->SetBinContent(i+1, fData(i)); // Populate data hist
      }
    }
  } // function LikelihoodCovMxExperiment::SaveHists

  void LikelihoodCovMxExperiment::EnableLMA() const {
    fUseLMA = true;
    for (size_t i = 0; i < fNBinsFull; ++i) fBeta(i) = 1.;
    MaskBetas();
  }

  void LikelihoodCovMxExperiment::ResetLambda() const {
    fLambda = fLambdaZero;
  }

  void LikelihoodCovMxExperiment::DecreaseLambda() const {
    fLambda /= fNu;
  }

  void LikelihoodCovMxExperiment::IncreaseLambda() const {
    fLambda *= fNu;
    if (fLambda > fLambdaZero) ResetLambda();
  }

}
