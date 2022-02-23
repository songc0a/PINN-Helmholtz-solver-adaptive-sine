function [spec,f]= fftrl(s,t,percent,n)
% FFTRL: forward Fourier transform for real-valued signals.
%
% [spec,f]= fftrl(s,t,percent,n)
% [spec,f]= fftrl(s,t,percent)
% [spec,f]= fftrl(s,t)
%
% Forward fourier transform of a real-valued signal. Relative to
% MATLAB's fft it returns only the positive frequencies in an array
% roughly half the size. If there are n input time samples, then there
% will be floor(n/2+1) frequency samples. This means that if n is an even
% number, then a time series of length n and one of length n+1 will produce
% frequency spectra of the same length. However, only the first case will
% have a sample at Nyquist.
% If the input trace is a vector, then the return
% is simply the transform of that vector. If a matrix, then each 
% column of the matrix is transformed. The inverse is accomplished by
% IFFTRL.
%
% s= input signal (trace or gather, i.e. vector or matrix) 
% t= input time coordinate vector
% percent= specifies the length of a raised cosine taper to be
%          applied to s (both ends) prior to any padding. 
%          Taper is a percent of the length of s. Taper is 
%          applied using MWINDOW.
%		********** Default=0% ***********
% n= length to which the input trace is to be padded with zeros.
%     ********** Default is the input length (no pad) ***********
%  
% spec= output spectrum
% f= output frequency sample vector
%
% by G.F. Margrave, May 1991, updated June 2004
%
% NOTE: It is illegal for you to use this software for a purpose other
% than non-profit education or research UNLESS you are employed by a CREWES
% Project sponsor. By using this software, you are agreeing to the terms
% detailed in this software's Matlab source file.
 
% BEGIN TERMS OF USE LICENSE
%
% This SOFTWARE is maintained by the CREWES Project at the Department
% of Geology and Geophysics of the University of Calgary, Calgary,
% Alberta, Canada.  The copyright and ownership is jointly held by 
% its author (identified above) and the CREWES Project.  The CREWES 
% project may be contacted via email at:  crewesinfo@crewes.org
% 
% The term 'SOFTWARE' refers to the Matlab source code, translations to
% any other computer language, or object code
%
% Terms of use of this SOFTWARE
%
% 1) Use of this SOFTWARE by any for-profit commercial organization is
%    expressly forbidden unless said organization is a CREWES Project
%    Sponsor.
%
% 2) A CREWES Project sponsor may use this SOFTWARE under the terms of the 
%    CREWES Project Sponsorship agreement.
%
% 3) A student or employee of a non-profit educational institution may 
%    use this SOFTWARE subject to the following terms and conditions:
%    - this SOFTWARE is for teaching or research purposes only.
%    - this SOFTWARE may be distributed to other students or researchers 
%      provided that these license terms are included.
%    - reselling the SOFTWARE, or including it or any portion of it, in any
%      software that will be resold is expressly forbidden.
%    - transfering the SOFTWARE in any form to a commercial firm or any 
%      other for-profit organization is expressly forbidden.
%
% END TERMS OF USE LICENSE
 
% set defaults
 if(nargin<4)
		n=length(t);
 end
 if(nargin<3)
   percent=0.0;
 end
% determine number of traces in ensemble
 [l,m]=size(s);
 ntraces=1;
 itr=0; %transpose flag
 if(l==1) nsamps=m; itr=1; s=s(:); %switch to column vectors
 elseif(m==1) nsamps=l;
 else
 	nsamps=l; ntraces=m;
 end
 if(nsamps~=length(t))
		t=t(1)+(t(2)-t(1))*(0:nsamps-1);
		 if(nargin<4)
			n=length(t);
		 end
	%error(' time vector and trace matrix don''t match in length');
 end
 
% apply the taper
 if(percent>0)
	 mw=mwindow(nsamps,percent);
	 mw=mw(:,ones(1,ntraces));
	 s=s.*mw;
	 clear mw
 end
% pad s if needed
 if (nsamps<n),
	s=[s;zeros(n-nsamps,ntraces)];
  	nsamps=n; 
 end

% transform the array
	spec=fft(s,nsamps);
	spec=spec(1:floor(n/2+1),:);% save only the positive frequencies
	clear s;
    
% build the frequency vector
fnyq=1. / (2*(t(2)-t(1)));
 nf=size(spec,1);
 df=2*fnyq/n;
 %f=linspace(0.,fnyq,nf)';
 f=df*(0:nf-1)';
 
 if(itr)
 	f=f';
 	spec=spec.';
 end
