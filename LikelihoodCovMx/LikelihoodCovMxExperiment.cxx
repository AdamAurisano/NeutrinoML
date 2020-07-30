#include "CAFAna/Experiment/LikelihoodCovMxExperiment.h"

#include "CAFAna/Core/HistCache.h"
#include "CAFAna/Core/LoadFromFile.h"
#include "CAFAna/Core/Utilities.h"

#include "OscLib/func/IOscCalculator.h"

#include "TDirectory.h"
#include "TObjString.h"
#include "TH1.h"
// #include "TMatrixDEigen.h"

#include "TCanvas.h" // delete me
#include "TLatex.h"  // delete me
#include "TGraph.h"  // delete me

#include <iomanip>
#include <iostream>
#include <chrono>

// include LAPACK routines
extern "C" {
  extern void dpotrf_(char*,int*,double*,int*,int*);
  extern void dpotrs_(char*,int*,int*,double*,int*,double*,int*,int*);
}

// #include <Eigen/SVD>
// #include <Eigen/Cholesky>

namespace ana
{
  // Standard C++ namespace
  using std::vector;
  using std::string;
  using std::count;
  using std::cout;
  using std::endl;
  using std::runtime_error;

  // Standard C++ chrono namespace
  using std::chrono::high_resolution_clock;
  using std::chrono::duration_cast;
  using std::chrono::seconds;
  using std::chrono::minutes;

  // Eigen tools
  // using Eigen::MatrixXd;

  // CAFAna covariance matrix namespace
  using covmx::Component;
  using covmx::Sample;
  using covmx::CovarianceMatrix;

  //-------Default constructor-------------------------------------------
  // LikelihoodCovMxExperiment::LikelihoodCovMxExperiment() {}
  
  //----------------------------------------------------------------------
  LikelihoodCovMxExperiment::LikelihoodCovMxExperiment(vector<Sample> samples,
    CovarianceMatrix* covmx, double epsilon, double lambdazero, double nu)
  : fSamples(samples), fCovMx(covmx), fNBins(0), fNBinsFull(0), fLambdaZero(lambdazero),
    fNu(nu), fLLUseROOT(false), fLLUseSVD(false), fBetaHist(nullptr), fMuHist(nullptr),
    fBetaMuHist(nullptr), fPredShiftedHist(nullptr), fPredUnshiftedHist(nullptr), fDataHist(nullptr),
    fMxInv(nullptr), fShapeOnly(false), fVerbose(false), fInversionTimeHist(nullptr)
  {
    for (Sample s: fSamples) {
      fNBins += s.GetBinning().NBins();
      fNBinsFull += s.GetBinning().NBins() * GetComponents(s).size();
      fNBinsPerSample.push_back(s.GetBinning().NBins());
      fComps.push_back(GetComponents(s));
    }

    if (fCovMx) {
      // Invert matrix ahead of time
      TMatrixD matrix = CovarianceMatrix::ForcePosDef(fCovMx->GetFullCovMx(), epsilon);
      // fMxInv = CovarianceMatrix::ForcePosDef(fCovMx->GetFullCovMx(), epsilon).invert();
      // Use Eigen to SVD decompose and invert
      TDecompSVD decomp(matrix);
      decomp.SetTol(1e-40);
      fMxInv = new TMatrixD(decomp.Invert());

      // Calculate M M^-1 - 1
      TDecompSVD closureDecomp(*fMxInv);
      closureDecomp.SetTol(1e-40);
      TMatrixD residualMx(fNBinsFull, fNBinsFull);
      residualMx.Mult(matrix, *fMxInv);

      for (size_t i = 0; i < fNBinsFull; ++i) residualMx(i,i) -= 1;

      fResidual = 0;
      for (size_t i = 0; i < fNBinsFull; ++i) {
        for (size_t j = 0; j < fNBinsFull; ++j) {
	  // Getting the absolute value of the biggest element of MM^-1 - 1
          double res = std::abs(residualMx(i,j));
          if (res >= fResidual) fResidual = res;
        }
      }
    }

    // Set up beta and mu
    fMu.resize(fNBinsFull);
    fBeta.resize(fNBinsFull);
    fBetaMask.resize(fNBinsFull);

    // Fill the data vector
    fData.resize(fNBins, 0);
    fCosmic.resize(fNBins, 0);
    size_t i = 0;
    for (size_t iS = 0; iS < fSamples.size(); ++iS) {
      double pot      = fSamples[iS].GetData()->POT();
      double livetime = fSamples[iS].GetData()->Livetime();
      TH1D* hData     = fSamples[iS].GetData()->ToTH1(pot);
      TH1D* hCos      = fSamples[iS].HasCosmic() ?
        fSamples[iS].GetCosmic()->ToTH1(livetime, kLivetime) : nullptr;
      for (size_t iBin = 1; iBin <= fNBinsPerSample[iS]; ++iBin) {
        fData[i] = hData->GetBinContent(iBin);
        fCosmic[i] = hCos ? hCos->GetBinContent(iBin) : 0;
        ++i;
      } // for bin i
      HistCache::Delete(hData);
      HistCache::Delete(hCos);
    } // for sample i
  }

  //----------------------------------------------------------------------
  LikelihoodCovMxExperiment::~LikelihoodCovMxExperiment()
  {
    if (fMxInv) delete fMxInv;
  }

  //----------------------------------------------------------------------
  double LikelihoodCovMxExperiment::ChiSq(osc::IOscCalculatorAdjustable* osc,
    const SystShifts& syst) const
  {
    DontAddDirectory guard;

    // Get MC and data spectra
    // fMu(fNBinsFull);
    size_t i = 0;
    for (size_t iS = 0; iS < fSamples.size(); ++iS) {
      double pot = fSamples[iS].GetData()->POT();
      SystShifts shift = fSamples[i].GetSystShifts(syst);
      for (Component comp: fComps[iS]) {
        Spectrum spec = fSamples[iS].GetPrediction()->PredictComponentSyst(osc,
    shift, std::get<0>(comp), std::get<1>(comp), std::get<2>(comp));
        TH1D* hMu = spec.ToTH1(pot);
        for (size_t iBin = 1; iBin <= (size_t)hMu->GetNbinsX(); ++iBin) {
    fMu[i] = hMu->GetBinContent(iBin);
    ++i;
        } // for bin
        HistCache::Delete(hMu);
      } // for component
    } // for sample
    for (size_t i = 0; i < fNBinsFull; ++i) fBeta[i] = 1.;

    // Do stats-only calculation if no matrix
    if (!fCovMx) {
      double ret = 0;
      vector<double> e = GetExpectedSpectrum();
      for (size_t i = 0; i < fData.size(); ++i) {
        ret += LogLikelihood(e[i], fData[i]);
      } // for bin i
      return ret;
    } // stats-only calculation

    // Number of bins
    if (fCovMx && fNBins != (size_t)fCovMx->GetBinning().NBins()) {
      throw runtime_error(
        "Number of bins in predictions does not match covariance matrix!");
    }

    // Fill histogram before beta shifting
    if (fPredUnshiftedHist) {
      vector<double> e = GetExpectedSpectrum();
      for (size_t j = 0; j < fNBins; ++j) {
        fPredUnshiftedHist->SetBinContent(j+1, e[j]);
      }
    } // if filling unshifted spectrum

    return LikelihoodCovMxNewton();

  } // function LikelihoodCovMxExperiment::ChiSq

  //---------------------------------------------------------------------------
  vector<double> LikelihoodCovMxExperiment::GetExpectedSpectrum() const {

    vector<double> ret(fNBins, 0.);
    unsigned int fullOffset(0), scaledOffset(0);

    for (size_t iS = 0; iS < fSamples.size(); ++iS) { // loop over samples
      for (size_t iC = 0; iC < fComps[iS].size(); ++iC) { // loop over components
        for (size_t iB = 0; iB < fNBinsPerSample[iS]; ++iB) { // loop over bins
    unsigned int iScaled = scaledOffset + iB;
    unsigned int iFull = fullOffset + (iC*fNBinsPerSample[iS]) + iB;
    ret[iScaled] += fMu[iFull]*fBeta[iFull];
        } // for bin
      } // for component
      // Now loop one more time to add cosmics
      for (size_t iB = 0; iB < fNBinsPerSample[iS]; ++iB) {
        unsigned int iScaled = scaledOffset + iB;
        ret[iScaled] += fCosmic[iScaled];
      } // for bin
      scaledOffset += fNBinsPerSample[iS]; // increase offset in scaled matrix
      fullOffset += fComps[iS].size() * fNBinsPerSample[iS];
    } // for sample
    return ret;

  } // function LikelihoodCovMxExperiment::GetExpectedSpectrum

  //---------------------------------------------------------------------------
  void LikelihoodCovMxExperiment::InitialiseBetas() const{
    double eps(1e-40);
    unsigned int fullOffset(0), scaledOffset(0);
    for (size_t iS = 0; iS < fSamples.size(); ++iS) { // loop over samples
      for (size_t iB = 0; iB < fNBinsPerSample[iS]; ++iB) { // loop over bins
        unsigned int iScaled = scaledOffset+iB;
        double sum = 0;
        for (size_t iC = 0; iC < fComps[iS].size(); ++iC) { // loop over components
    unsigned int iFull = fullOffset+(iC*fNBinsPerSample[iS])+iB;
    sum += fMu[iFull];
        } // for component
        double val = fData[iScaled] / (sum + eps);
        if (val < 0) val = 0;
        for (size_t iC = 0; iC < fComps[iS].size(); ++iC) { // loop over components
    unsigned int iFull = fullOffset+(iC*fNBinsPerSample[iS])+iB;
    fBeta[iFull] = val;
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
    double y(0.); // calculate y
    for (size_t jC = 0; jC < fComps[iS].size(); ++jC) {
      if (iC != jC) { // beta mu from this bin is 0
        unsigned int jFull = iFullOffset+(jC*fNBinsPerSample[iS])+iB;
        y += fMu[jFull] * fBeta[jFull];
      }
    } // for component j
    if (y < 1e-40) y = 1e-40; // clamp y
    double z(0.); // calculate z
    for (size_t jFull = 0; jFull < fNBinsFull; ++jFull) {
      double beta = (jFull == iFull) ? 0 : fBeta[jFull]; // set this beta to 0
      z += (beta-1)*(*fMxInv)(iFull, jFull);
    }

    // Calculate gradient
    double gradZero = 2 * (fMu[iFull]*(1.0-(fData[iScaled]/y)) + z);
    if (gradZero > 0 && fBetaMask[iFull]) { // if we're masking a bin off
      // cout << "Masking off bin " << iFull+1 << endl;
      fBetaMask[iFull] = false;
      maskChange = true;
    }
    if (gradZero <= 0 && !fBetaMask[iFull]) { // if we're masking a bin on
      // cout << "Masking on bin " << iFull+1 << endl;
      fBetaMask[iFull] = true;
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
    double y(0.); // calculate y
    for (size_t jC = 0; jC < fComps[iS].size(); ++jC) {
      unsigned int jFull = iFullOffset+(jC*fNBinsPerSample[iS])+iB;
      y += fMu[jFull] * fBeta[jFull];
    } // for component j
    if (y < 1e-40) y = 1e-40; // clamp y
    double z(0.); // calculate z
    for (size_t jFull = 0; jFull < fNBinsFull; ++jFull) {
      z += (fBeta[jFull]-1)*(*fMxInv)(iFull, jFull);
    }

    // Calculate gradient
    fGrad[iFull] = 2 * (fMu[iFull]*(1.0-(fData[iScaled]/y)) + z);
    // cout << "grad is " << grad[iFull] << endl;
    // cout << "Bin " << iFull+1 << ": gradient is " << grad[iFull] <<", data is "
      // << fData[iScaled] << ", y is " << y << ", z is " << z << endl;

    // Populate Hessian matrix
    size_t jFullOffset(0.), jScaledOffset(0.);
    for (size_t jS = 0; jS < fSamples.size(); ++jS) { // loop over samples
      for (size_t jC = 0; jC < fComps[jS].size(); ++jC) { // loop over components
        for (size_t jB = 0; jB < fNBinsPerSample[jS]; ++jB) { // loop over bins
    // Get the current scaled & full bin
    unsigned int jScaled = jScaledOffset + jB;
    unsigned int jFull = jFullOffset+(jC*fNBinsPerSample[jS])+jB;
    unsigned int iHess = (jFull*fNBinsFull) + iFull;
    if (iScaled != jScaled) {
      fHess[iHess] = 2 * (*fMxInv)(iFull, jFull);
    } else {
      fHess[iHess] = 2 * (fMu[iFull]*fMu[jFull]*(fData[iScaled]/(y*y)) + (*fMxInv)(iFull, jFull));
      // if (iFull == jFull) cout << "Diagonal Hessian value: " << hess[iHess] << endl;
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
      fHess[(fNBinsFull+1)*i] *= 1. + fLambda;
      fGrad[i] *= -1;
    }

  } // function LikelihoodCovMxExperiment::GetGradAndHess

  //---------------------------------------------------------------------------
  void LikelihoodCovMxExperiment::GetReducedGradAndHess() const {

    // Mask off bins
    size_t maskedSize = count(fBetaMask.begin(), fBetaMask.end(), true);

    GetGradAndHess();

    fGradReduced = new double[maskedSize];
    fHessReduced = new double[maskedSize*maskedSize];

    size_t c = 0;
    for (size_t i = 0; i < fNBinsFull; ++i) {
      if (fBetaMask[i]) {
        fGradReduced[c] = fGrad[i];
        ++c;
      }
    }
    c = 0;
    for (size_t i = 0; i < fNBinsFull; ++i) {
      for (size_t j = 0; j < fNBinsFull; ++j) {
        size_t index = (i * fNBinsFull) + j;
        if (fBetaMask[i] && fBetaMask[j]) {
          fHessReduced[c] = fHess[index];
          ++c;
        }
      }
    }
    
  } // function LikelihoodCovMxExperiment::GetReducedGradAndHess

  //---------------------------------------------------------------------------
  void LikelihoodCovMxExperiment::Solve(int N, double* hess,
    double* grad) const {

    int INFO;
    string L ="L";
    int NRHS = 1; // Size of right-hand-side solution

    // Use LAPACK to Cholesky factorise the Hessian matrix
    dpotrf_(&L[0],&N,hess,&N,&INFO);
    // If Cholesky factorisation succeeded, solve for pulls
    if (INFO == 0) dpotrs_(&L[0],&N,&NRHS,hess,&N,grad,&N,&INFO);

    if (INFO != 0) { // If either step failed, throw an error
      std::ostringstream err;
      err << "Cholesky factorisation failed at matrix element " << INFO;
      throw std::runtime_error(err.str());
    }
  } // function LikelihoodCovMxExperiment::Solve

  //---------------------------------------------------------------------------
  double LikelihoodCovMxExperiment::LikelihoodCovMxNewton() const {

    auto start = high_resolution_clock::now();

    fGrad = new double[fNBinsFull];
    fHess = new double[fNBinsFull*fNBinsFull];

    vector<double> e = GetExpectedSpectrum();

    for (size_t i = 0; i < fNBinsFull; ++i) fBetaMask[i] = true;

    // We're trying to solve for the best expectation in each bin and each component. A good seed
    // value is 1 (no shift).
    // Note: beta will contain the best fit systematic shifts at the end of the process.
    // This should be saved to show true post-fit agreement
    double prev = -999;//, last = -999;//, prevprev = -999, best = -999;
    const int maxIters = 1e7;
    fIteration = 0;
    InitialiseBetas();
    MaskBetas();
    fLambda = fLambdaZero; // Initialise lambda at starting value

    TFile* f = TFile::Open("likelihood_iteration_plots.root", "recreate");
    TCanvas c("c", "c", 1600, 900);
    c.SetLogy(true);
    TGraph g1;

    for (size_t i = 0; i < fNBins; ++i) fPredShiftedHist->SetBinContent(i+1, e[i]);
    fPredShiftedHist->SetLineColor(kGray+1);
    fPredShiftedHist->SetLineWidth(2);
    fDataHist->Draw("e0");
    fPredShiftedHist->Draw("hist same");
    fDataHist->Draw("e0 same");
    f->WriteTObject(&c, Form("Iteration %d", fIteration));

    while (true) {
      ++fIteration;
      if (fIteration > maxIters) {

        std::ostringstream err;
        err << "No convergence after " << maxIters << " iterations! Quitting out of the infinite loop.";
        throw runtime_error(err.str());
      }

      // GetGradAndHess();

      // for (size_t i = 0; i < fNBinsFull; ++i) {
        // if (fBetaMask[i] && fGrad[i] > 0) fBetaMask[i] = false;
      // }

      // vector<double> newBeta(fNBinsFull);

      // If we have negative betas, mask them out
      unsigned int N = count(fBetaMask.begin(), fBetaMask.end(), true);
      GetReducedGradAndHess();
      Solve(N, fHessReduced, fGradReduced);
      size_t counter = 0;
      for (size_t i = 0; i < fNBinsFull; ++i) {
        if (fBetaMask[i]) {
          fBeta[i] += fGradReduced[counter];
          if (fBeta[i] < 0) { // if we pulled negative, mask this beta off
            fBeta[i] = 0; 
            fBetaMask[i] = false;
          }
          ++counter;
        } else {
          fBeta[i] = 0;
        }
      } // for bin
      delete fGradReduced;
      delete fHessReduced;
        // newBeta[i] = fBeta[i];
        // if (fBetaMask[i]) {
          // newBeta[i] += fGradReduced[c]; // Update betas
          // ++c;
        // }
        // if (newBeta[i] < 0) { // If we have negative pulls...
          // cout << "Masking off beta " << i+1 << endl;
          // fBetaMask[i] = false; // ...mask them out...
          // newBeta[i] = 0; // ...set them to zero...
          // maskChange = true; // ...and iterate again.
          // cout << "Masking off beta " << i+1 << endl;
        // }

      // bool maskChange = true;
      // while (maskChange) { // Keep reducing the matrix until we have no negative pulls
      //   // cout << "Iterating to mask off bins..." << endl;
      //   maskChange = false;
      //   GetReducedGradAndHess(); // Cut out any masked bins
      //   Solve(N, fHessReduced, fGradReduced);
      //   size_t c = 0;
      //   for (size_t i = 0; i < fNBinsFull; ++i) {
      //     newBeta[i] = fBeta[i];
      //     if (fBetaMask[i]) {
      //       newBeta[i] += fGradReduced[c]; // Update betas
      //       ++c;
      //     }
      //     if (newBeta[i] < 0) { // If we have negative pulls...
      //       // cout << "Masking off beta " << i+1 << endl;
      //       fBetaMask[i] = false; // ...mask them out...
      //       newBeta[i] = 0; // ...set them to zero...
      //       maskChange = true; // ...and iterate again.
      //       // cout << "Masking off beta " << i+1 << endl;
      //     }
      //   } // for bin i
        // N = count(fBetaMask.begin(), fBetaMask.end(), true);

      // Update betas
      // cout << "Updating betas" << endl << endl;
      // for (size_t i = 0; i < fNBinsFull; ++i) fBeta[i] = newBeta[i];

      // Predict collapsed spectrum
      e = GetExpectedSpectrum();

      // Update the chisq
      // There's the LL of the data to the updated prediction...
      fStatChiSq = 0;
      for(unsigned int i = 0; i < fNBins; ++i) {
        fStatChiSq += LogLikelihood(e[i], fData[i]);
      }

      // ...plus the penalty the prediction picks up from its covariance
      fSystChiSq = 0;
      for(unsigned int r = 0; r < fNBinsFull; ++r) {
        for(unsigned int t = 0; t < fNBinsFull; ++t) {
          fSystChiSq += (fBeta[t]-1) * (*fMxInv)(t, r) * (fBeta[r]-1);
        }
      }
      double chisq = fStatChiSq + fSystChiSq;

      for (size_t i = 0; i < fNBins; ++i) fPredShiftedHist->SetBinContent(i+1, e[i]);
      fDataHist->Draw("e0");
      fPredShiftedHist->Draw("hist same");
      fDataHist->Draw("e0 same");
      f->WriteTObject(&c, Form("Iteration %d", fIteration));

      // if (!quadratic && fIteration > 1 && chisq < prev) fLambda /= fNu;
      if (fIteration > 1 && chisq < prev) fLambda /= fNu;

      g1.SetPoint(fIteration, fIteration, chisq);

      // If the updates didn't change anything at all then we're done
      if (fabs(chisq-prev) < 1e-10) {
        if (MaskBetas()) { // re-mask betas - if the mask changed, go again
          // while (MaskChange()) continue;
        // if (N < fNBinsFull && fabs(chisq-last) > 1e-10) {
          fLambda = 0.1;
          // last = chisq;
          // for (size_t i = 0; i < fNBinsFull; ++i) fBetaMask[i] = true;
        }

        else {

          cout << "Converged to " << chisq << " (" << fStatChiSq << " stat, " << fSystChiSq
            << " syst) after " << fIteration << " iterations and ";
          auto end = high_resolution_clock::now();
          double secs = duration_cast<seconds>(end-start).count();
          if (secs < 60) cout << secs << " seconds." << endl;
          else {
            double mins = duration_cast<minutes>(end-start).count();
            cout << mins << " minutes." << endl;
          }

          // cout << "HOW MANY POINTS?????? " << g.GetN() << endl;

          g1.Draw("AC");
          c.SetLogy(true);
          g1.SetLineWidth(2);
          c.SaveAs("chisq_fixed.pdf");
          // f->WriteTObject(&c, "iterations");

          // g2.Draw("AC");
          // c.SetLogy(true);
          // c.SaveAs("chisq2.pdf");

          // Fill histograms if necessary
          if (fBetaHist)        for (size_t i = 0; i < fNBinsFull; ++i) fBetaHist->SetBinContent(i+1, fBeta[i]);
          if (fMuHist)          for (size_t i = 0; i < fNBinsFull; ++i) fMuHist->SetBinContent(i+1, fMu[i]);
          if (fBetaMuHist)      for (size_t i = 0; i < fNBinsFull; ++i) fBetaMuHist->SetBinContent(i+1, fBeta[i]*fMu[i]);
          if (fPredShiftedHist) for (size_t i = 0; i < fNBins; ++i) fPredShiftedHist->SetBinContent(i+1, e[i]);
  	
          delete fGrad;
          delete fHess;
          delete f;
          // abort();
          return chisq;
        }
      }

      prev = chisq;

    } // end while
    
  } // function LikelihoodCovMxExperiment::LogLikelihoodCovMxNewton

  //----------------------------------------------------------------------
  void LikelihoodCovMxExperiment::SaveHists(bool opt) {
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
      fBetaHist          = HistCache::New(";;", fullBins);
      fMuHist            = HistCache::New(";;", fullBins);
      fBetaMuHist        = HistCache::New(";;", fullBins);
      fPredShiftedHist   = HistCache::New(";;", scaledBins);
      fPredUnshiftedHist = HistCache::New(";;", scaledBins);
      fDataHist          = HistCache::New(";;", scaledBins);
      for (size_t i = 0; i < fData.size(); ++i) {
        fDataHist->SetBinContent(i+1, fData[i]); // Populate data hist
      }
    }
  } // function LikelihoodCovMxExperiment::SaveHists

  //----------------------------------------------------------------------
  TH1I* LikelihoodCovMxExperiment::InversionTimeHist()
  {
    if( !fInversionTimeHist ){
      std::cerr << "Inversion Time histogram not configured to fill, "
      << "use LikelihoodCovMxExperiment::SetInversionTimeHist() before running the fit"
      << endl;
      abort();
    }
    return fInversionTimeHist;
  }
  //----------------------------------------------------------------------
  void LikelihoodCovMxExperiment::SetInversionTimeHist(bool timeit, int tmax, int tmin)
  {
    if( !timeit ){
      fInversionTimeHist = nullptr;
    } else {
      fInversionTimeHist = new TH1I("inversion_time",
        ";Single Inversion Time (ms);Number of Inversions",
        1000, tmin, tmax);
    }
  }

}
