#pragma once

//#include "OscLib/func/OscCalculatorSterile.h"

#include "CAFAna/Experiment/IChiSqExperiment.h"
#include "CAFAna/Prediction/IPrediction.h"
#include "CAFAna/Core/Spectrum.h"
#include "CAFAna/Prediction/CovarianceMatrix.h"

#include "TMatrixD.h"

#include "TDecompSVD.h"

namespace ana
{
  /// Compare a single data spectrum to the MC + cosmics expectation
  class LikelihoodCovMxExperiment: public IChiSqExperiment
  {
  public:
    /// \param samples Source of concatenated oscillated MC beam predictions
    /// \param covgen Source of predicted covariance matrix
    /// \epsilon epsilon value to add onto the diagonal to aid matrix inversion
    // Default contructor
    // LikelihoodCovMxExperiment();
    LikelihoodCovMxExperiment(std::vector<covmx::Sample> samples,
          covmx::CovarianceMatrix* covmx, double epsilon=0.1,
          double lambdazero=0.1, double nu=1.5);
    
    virtual ~LikelihoodCovMxExperiment();

    virtual double ChiSq(osc::IOscCalculatorAdjustable* osc,
      const SystShifts& syst = SystShifts::Nominal()) const;

    double GetStatChiSq() { return fStatChiSq; };
    double GetSystChiSq() { return fSystChiSq; };
    double GetResidual()  { return fResidual;  };
    double GetIteration() { return fIteration; };

    // Debug histograms
    void SaveHists(bool opt=true);
    TH1D* GetBetaHist()          { return fBetaHist; };
    TH1D* GetMuHist()            { return fMuHist; };
    TH1D* GetBetaMuHist()        { return fBetaMuHist; };
    TH1D* GetPredShiftedHist()   { return fPredShiftedHist; };
    TH1D* GetPredUnshiftedHist() { return fPredUnshiftedHist; };
    TH1D* GetDataHist()          { return fDataHist; };

    // Configurable options
    void SetVerbose(bool opt)          { fVerbose = opt; };
    void LLUseROOT(bool val=true)      { fLLUseROOT = val; };
    void LLUseSVD(bool val=true)       { fLLUseSVD = val; };
    void SetShapeOnly(bool val = true) { fShapeOnly = val; };

    TH1I* InversionTimeHist();
    /// \param timeit      Whether or not to save inversion time in a histogram  
    /// \param tmax        Upper boundary on time hist in milliseconds
    /// \param tmin        Lower boundary on time hist in milliseconds
    void SetInversionTimeHist(bool timeit = true, int tmax = 1000, int tmin = 0); // milliseconds

  protected:

    // double Par2Beta(double par) const;
    // double Beta2Par(double beta) const;

    /// \param mu           Vector of expected events
    /// \param beta         Vector of shifted beta values
    std::vector<double> GetExpectedSpectrum() const;

    void InitialiseBetas() const;

    void GetGradAndHess() const;
    void GetReducedGradAndHess() const;

    void Solve(int N, double* hess, double* grad) const;
    double LikelihoodCovMxNewton() const;
    
    std::vector<covmx::Sample> fSamples;
    covmx::CovarianceMatrix* fCovMx;
    mutable std::vector<double> fMu;
    mutable std::vector<double> fBeta;
    mutable std::vector<bool> fBetaMask;
    std::vector<double> fData;
    std::vector<double> fCosmic;

    mutable double* fGrad;
    mutable double* fHess;
    mutable double* fGradReduced;
    mutable double* fHessReduced;

    unsigned int fNBins;
    unsigned int fNBinsFull;
    std::vector<unsigned int> fNBinsPerSample;
    std::vector<std::vector<covmx::Component>> fComps;

    double fLambdaZero;     /// Levenberg-Marquardt starting lambda
    double fNu;             /// Levenberg-Marquardt nu
    mutable double fLambda; /// Levenberg-Marquardt lambda

    TH1D* fBetaHist;
    TH1D* fMuHist;
    TH1D* fBetaMuHist;
    TH1D* fPredShiftedHist;
    TH1D* fPredUnshiftedHist;
    TH1D* fDataHist;
    TMatrixD* fMxInv;
    bool fShapeOnly;
    bool fVerbose;

    mutable double fStatChiSq;
    mutable double fSystChiSq;
    mutable double fResidual;
    mutable int    fIteration;

    TH1I* fInversionTimeHist;

  };
}
