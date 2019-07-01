function [ X, S ] = SIM_MRS( T, model, theta, S0)
%SIM_MRS simulates an MRS model
% Inputs,
%   - T, the length of the simulated data
%   - model, a cell array containing strings specifying the model.
%       eg 1. for an MRS model with one AR(1) base regime of type II, one spike regime
%       and one drop regime we would specify
%       model = {{'AR(1) type II'}{'LN shifted' 'LN shifted and reversed'}}
%       eg 2. for an MRS model with two AR(1) base regimes of type III, two spike
%       regimes and one drop regime we would specify
%       model = {{'AR(1) type III' 'AR(1) type III'}{'LN shifted' 'LN sifted' 'LN shifted and reversed'}}
%       Notes:
%       * Only put up to two 'AR(1)'s in the first element of model. We may
%         specify either type II, where the AR(1) regime evolves at all
%         time points, or type III, where the AR(1) regime evolves only
%         when observed.
%       * In the second element of model we can choose to specify up to
%         three of the following,
%           'G' ~ Normal distribution
%           'LN' ~ log-normal distribution
%           'LN shifted' ~ shifted lognormal distribution
%           'LN shifted and reversed' ~ shifted and reversed lognormal
%           distribution
%           'Gamma' ~ Shifted gamma distribution. Same shifting mechanism
%           as the shifted lognormal distribution.
%       * Where more than 1 'AR(1)' regime is specified the we resrict the
%         variances of each model, \sigma_1<\sigma_2<...
%       * We may only specify 2 'LN shifted' regimes, and when we do, we
%         restrict the shifting parameter if the first one to be less than
%         the shifting parameter of the second one.
%       * We may only specify 1 'LN shifted and reversed' regime.
%   - theta, a vector of paramters in the same order as
%     the model specification followed by the transition matrix.
%     i.e. for the model = {{'AR(1)'} {'LN shifted'}},
%     theta=[AR(1) parameter \alpha, AR(1) parameter \phi, AR(1) parameter \sigma, iid q, iid param \mu, iid \sigma, p_11,p_12,p_21,p_22]
%   - S0, an initial distribution of the hidden sequence S
% Outputs,
%   - X, a time series of the data  (column vector)
%   - S, the hidden regime sequence (column vector)

% simulate the hidden process
L = length(model{1})+length(model{2});
P = reshape(theta(end-L^2+1:end),L,L)';
P_PDF = cumsum(P,2);
S = zeros(T,1);
S0_PDF = cumsum(S0);
S(1) = find(rand<S0_PDF,1);
for t=2:T
    S(t) = find(rand<P_PDF(S(t-1),:),1);
end

% initialise the observaion process
X = zeros(T,1);
m = [model{1};model{2}];
idx = ((S(1)-1)*3+1):(S(1)*3);
params = theta(idx);
switch m{S(1)}
    case 'AR(1)'
        mu = params(1)/(1-params(2));
        s = sqrt(params(3)/(1-params(2)^2));
        X(1) = normrnd(mu,s,1);
    case 'AR(1) type II'
        mu = params(1)/(1-params(2));
        s = sqrt(params(3)/(1-params(2)^2));
        X(1) = normrnd(mu,s,1);
    case 'AR(1) type III'
        mu = params(1)/(1-params(2));
        s = sqrt(params(3)/(1-params(2)^2));
        X(1) = normrnd(mu,s,1);
    case 'G'
        mu = params(2);
        s = sqrt(params(3));
        X(1) = normrnd(mu,s,1);
    case 'LN'
        mu = params(2);
        s = sqrt(params(3));
        X(1) = lognrnd(mu,s,1);
    case 'Gamma'
        A = params(2);
        B = params(3);
        X(1) = gamrnd(A,B,1);
    case 'LN reversed'
        mu = params(2);
        s = sqrt(params(3));
        X(1) = -lognrnd(mu,s,1);
    case 'Gamma reversed'
        A = params(2);
        B = params(3);
        X(1) = -gamrnd(A,B,1);
end

% simulate the observation process
for t=2:T
    idx = ((S(t)-1)*3+1):(S(t)*3);
    params = theta(idx);
    switch m{S(t)}
        case 'AR(1)'
            mu = params(1) + params(2)*X(t-1);
            s = sqrt(params(3));
            X(t) = normrnd(mu,s,1);
        case 'AR(1) type II'
            t_1 = find(S(1:t-1)==S(t),1,'last'); 
            if isempty(t_1)
                mu = params(1)/(1-params(2));
                s = sqrt(params(3)/(1-params(2)^2));
            else
                lag = t-t_1;
                temp = sum(params(2).^(0:lag-1));
                mu = params(1)*temp + params(2).^lag*X(t_1);
                s = sqrt(params(3)*temp);
            end
            X(t) = normrnd(mu,s,1);
        case 'AR(1) type III'
            t_1 = find(S(1:t-1)==S(t),1,'last');
            if isempty(t_1)
                mu = params(1)/(1-params(2));
                s = sqrt(params(3)/(1-params(2)^2));
            else
                mu = params(1) + params(2)*X(t_1);
                s = sqrt(params(3));
            end
            X(t) = normrnd(mu,s,1);
        case 'G'
            mu = params(2);
            s = sqrt(params(3));
            X(t) = normrnd(mu,s,1);
        case 'LN'
            mu = params(2);
            s = sqrt(params(3));
            X(t) = lognrnd(mu,s,1);
        case 'Gamma'
            A = params(2);
            B = params(3);
            X(t) = gamrnd(A,B,1);
        case 'LN reversed'
            mu = params(2);
            s = sqrt(params(3));
            X(t) = -lognrnd(mu,s,1);
        case 'Gamma reversed'
            A = params(2);
            B = params(3);
            X(t) = -gamrnd(A,B,1);
    end
end

end











