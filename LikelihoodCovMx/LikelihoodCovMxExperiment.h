#pragma once

#include "CAFAna/Experiment/IExperiment.h"
#include "CAFAna/Prediction/IPrediction.h"
#include "CAFAna/Core/Spectrum.h"
#include "CAFAna/Prediction/CovarianceMatrix.h"

#include <Eigen/Core>

namespace ana
{
  /// Compare a single data spectrum to the MC + cosmics expectation
  class LikelihoodCovMxExperiment: public IExperiment
  {
  public:
    /// \param samples Source of concatenated oscillated MC beam predictions
    /// \param covmx Covariance matrix
    /// \epsilon epsilon value to add onto the diagonal to aid matrix inversion
    LikelihoodCovMxExperiment(std::vector<covmx::Sample> samples,
          covmx::CovarianceMatrix* covmx, double epsilon=1e-5,
          double lambdazero=0, double nu=10);
    
    virtual ~LikelihoodCovMxExperiment() {};

    double ChiSq(osc::IOscCalcAdjustable* osc,
      const SystShifts& syst = SystShifts::Nominal()) const override;

    void Reset() const override;

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
    void SetResetShifts(bool val = false) { fResetShifts = val; };

  protected:

    Eigen::ArrayXd GetExpectedSpectrum() const;
    double GetChiSq(Eigen::ArrayXd e) const;

    void InitialiseBetas() const;
    bool MaskBetas() const;
    void GetGradAndHess() const;
    void GetReducedGradAndHess() const;
    double LikelihoodCovMxNewton() const;
    void EnableLMA() const;
    void ResetLambda() const;
    void DecreaseLambda() const;
    void IncreaseLambda() const;
    
    std::vector<covmx::Sample> fSamples;
    covmx::CovarianceMatrix* fCovMx;
    mutable Eigen::ArrayXd fMu;
    mutable Eigen::ArrayXd fBeta;
    mutable std::vector<bool> fBetaMask;
    Eigen::ArrayXd fData;
    Eigen::ArrayXd fCosmic;

    mutable Eigen::VectorXd fGrad;
    mutable Eigen::MatrixXd fHess;
    mutable Eigen::VectorXd fGradReduced;
    mutable Eigen::MatrixXd fHessReduced;

    unsigned int fNBins;
    unsigned int fNBinsFull;
    std::vector<unsigned int> fNBinsPerSample;
    std::vector<std::vector<covmx::Component>> fComps;

    mutable bool fUseLMA;   /// Whether to use LMA
    double fLambdaZero;     /// Levenberg-Marquardt starting lambda
    double fNu;             /// Levenberg-Marquardt nu
    mutable double fLambda; /// Levenberg-Marquardt lambda

    TH1D* fBetaHist;
    TH1D* fMuHist;
    TH1D* fBetaMuHist;
    TH1D* fPredShiftedHist;
    TH1D* fPredUnshiftedHist;
    TH1D* fDataHist;
    Eigen::MatrixXd fMxInv;
    bool fResetShifts;
    bool fVerbose;

    mutable double fStatChiSq;
    mutable double fSystChiSq;
    mutable double fResidual;
    mutable int    fIteration;

  };
}
