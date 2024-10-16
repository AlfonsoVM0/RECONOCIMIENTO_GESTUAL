%function [cepstra, spectra, pspectrum, lpcas, F, M] = rastaplp(samples, sr, dorasta, modelorder)
function [cepstra, spectra, pspectrum, lpcas, F, M] = rastaplp(samples, sr, dorasta, modelorder, win_size, step)
%[cepstra, spectra, lpcas] = rastaplp(samples, sr, dorasta, modelorder)
%
% cheap version of log rasta with fixed parameters
%
% output is matrix of features, row = feature, col = frame
%
% sr is sampling rate of samples, defaults to 8000
% dorasta defaults to 1; if 0, just calculate PLP
% modelorder is order of PLP model, defaults to 8.  0 -> no PLP
%
% rastaplp(d, sr, 0, 12) is pretty close to the unix command line
% feacalc -dith -delta 0 -ras no -plp 12 -dom cep ...
% except during very quiet areas, where our approach of adding noise
% in the time domain is different from rasta's approach 
%
% 2003-04-12 dpwe@ee.columbia.edu after shire@icsi.berkeley.edu's version

if nargin < 2
  sr = 8000;
end
if nargin < 3
  dorasta = 1;
end
if nargin < 4
  modelorder = 8;
end

% add miniscule amount of noise
%samples = samples + randn(size(samples))*0.0001;

% first compute power spectrum
% pspectrum = powspec(samples, sr);

%fprintf('\nrastaplp\n')
%win_size
%step
wintime=double(win_size)/double(sr);
steptime=double(step)/double(sr);

pspectrum = powspec(samples, sr, wintime, steptime);

% next group to critical bands
aspectrum = audspec(pspectrum, sr,40);
nbands = size(aspectrum,1);

if dorasta ~= 0

% put in log domain
  nl_aspectrum = log(aspectrum);
  %nl_aspectrum=aspectrum;
  % next do rasta filtering  
  ras_nl_aspectrum = rastafilt(nl_aspectrum);
  % do inverse log
  aspectrum = exp(ras_nl_aspectrum);
  %aspectrum=ras_nl_aspectrum;

  

end  
  
% do final auditory compressions
%postspectrum = postaud(aspectrum, sr/2); % 2012-09-03 bug: was sr
postspectrum = aspectrum;

if modelorder > 0

  % LPC analysis 
  lpcas = dolpc(postspectrum, modelorder);

  % convert lpc to cepstra
  cepstra = lpc2cep(lpcas, modelorder+1);

  % .. or to spectra
  [spectra,F,M] = lpc2spec(lpcas, nbands);

else
 
  % No LPC smoothing of spectrum
  spectra = postspectrum;
  cepstra = spec2cep(spectra);
  
end

cepstra = lifter(cepstra, 0.6);

