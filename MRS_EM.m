function [ likelihood, MLEs, si, fi, predi, its_count, tol, Err ] = MRS_EM( data, model, theta, varargin )
%MRS_EM uses the EM algorithm to find the MLEs od MRS models with
%independent regimes for electricity prices. \cite{my thesis}
% Inputs
%   - data, a column vector of observed electricity prices
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
%   - theta, a (optional) vector of initial paramters in the same order as
%     the model specification followed by the transition matrix.
%     i.e. for the model = {{'AR(1)'} {'LN shifted'}},
%     theta=[AR(1) parameter \alpha, AR(1) parameter \phi, AR(1) parameter \sigma, iid q, iid param \mu, iid \sigma, p_11,p_12,p_21,p_22]
%
% Outputs
%   - MLEs, the MLEs in the same order as the optional input, theta
%     i.e. for the model = {{'AR(1)'} {'LN shifted'}},
%     theta=[AR(1) parameter \alpha, AR(1) parameter \phi, AR(1) parameter \sigma, iid q, iid param \mu, iid \sigma, p_11,p_12,p_21,p_22]
%     when no shifting parameter is in the model the space with q in it is
%     filled by 0 instead.
%   - si, the smoothed infernces P(R_t=i|x_{0:T}, MLEs)
%   - fi, the filtered infernces P(R_t=i|x_{0:T=t}, MLEs)
%   - predi, the prediction infernces P(R_{t+1}=i|x_{0:T}, MLEs)
%
%
% Angus Lewis, 6 June 18.

data = data(:);
l1 = length(model{1});
l2 = length(model{2});
L = l1+l2;
T = length(data);
MIN = [quantile(data,2/3),quantile(data,0.98)];
MAX = max(data);
options = optimoptions('fmincon','Display','off'); % if fmincon is used for the E-step, this stops the display showing

if nargin <3 % if no initial point is given for EM, allocate one
    theta = [repmat([0,0.5,1],1,l1), repmat([0,1,1],1,l2),repmat(repmat(1/L,1,L),1,L)];
end
if length(theta)~=3*l1+3*l2+L^2
    ME = MException('MATLAB:MRS_EM:InputErrortheta','theta is not the right length');
    throw(ME)
end
if any( abs(sum(reshape(theta( (end-(L^2-1)):end ),L,L)',2)-ones(L,1))>1e-3 )
    P = reshape(theta( (end-(L^2-1)):end ),L,L)'
    ME = MException('MATLAB:MRS_EM:InputErrorP','Row sums of P are not 1');
    throw(ME)
end
trunc = T;
bw = 0;
display = 1; 
for var_arg = 1:length(varargin)
    if strcmp(varargin{var_arg},'loglikelihood')
        bw = 1;
    end
    
    if strcmp(varargin{var_arg},'truncate')
        trunc = min(varargin{var_arg+1},T)+1;
    end
    
    if strcmp(varargin{var_arg},'display')
        if strcmp(varargin{var_arg+1},'off')
            display = 0; 
        end
    end
end
trunc = min(trunc,T);

p0 = (1/L)*ones(1,L); % this is the initial distribution across states, P(R_0=i)

theta_last_iteration = theta;
ll_old = Inf;
ll_tol = Inf;
tol = inf(1,10);
maxits = 1000;
its_count = 0;

Err = 'none';
Y=sort(data);

if bw==0
    while (any(tol > sqrt(eps)*10) || ll_tol > sqrt(eps)*10)
        
        %% E-step
        % run the forward algorithm and save the filtered inferences,
        % prediction inferences, and the likelihood. These are used in the
        % backward algorithm, and to calculate updates for P
        [fi,predi,likelihood] = forward(data, theta, p0, model, trunc,bw);%likelihood, show=predi(:,1,:); permute(show,[1,3,2]),data,theta(4),
        if its_count == maxits
            break
        end
        its_count = its_count + 1;
        % run the backward algorithm and save the smoothed inferences. These
        % are used to find the maximisers of Q
        P = reshape(theta( (end-(L^2-1)):end ),L,L)';
        si = backward(fi,predi,P,l1,T,trunc);
        
        idx = [{1} {':'} repmat({1},1,l1)];
        p0 = si(idx{:});
        %% M-step
        %% P matrix updates
        P_hat = zeros(size(P));
        for i = 1:l1 % when R_{t-1}=i is an AR(1) regime
            idx = [repmat({':'},1,i-1),1,repmat({':'},1,l1-i)]; % an index that tells us where N_{t,i}=1
            temp_i = si(1:T-1,i,:); % get the smoothed probabilities R_t = i for t = 1,...,T-1
            s_i = sum(temp_i(:)); % add them up to get \sum_{t=1}^{T-1}  P( R_t=i | x_{0:T} )
            for r = 1:L
                temp_r = si(2:T,r,idx{:}); % get the smoothed probabilities with N_{t,i} = 1 (which means R_{t-1}=i) and R_t=r
                P_hat(i,r) = sum(temp_r(:))/s_i; % summing temp_r gives P( R_t=r, N_{t,i}=1 | x_{0:T} ) = P( R_t=r, R_{t-1}=i | x_{0:T} )
            end
        end
        
        % when R_{t-1}=i is NOT an AR(1) regime
        s = zeros(L-l1,L);
        rm_mt_from_1 = repmat({1:trunc},1,l1);
        rm_mt_from_2 = repmat({[2:trunc,trunc]},1,l1);
        for t = 2:T
            mt = min(t,trunc);
            if mt == trunc
                map_from_1 = rm_mt_from_1;
                map_from_2 = rm_mt_from_2;
            else
                map_from_1 = repmat({1:t},1,l1); % the indices where fi(t-1,:) has non-zero values
                map_from_2 = repmat({2:t+1},1,l1); % the indices with N_{t,i} > 2 for all i
            end
            %fi(t-1,l1+1:L,map_from_1{:}); % P( N_{t-1}=n-1, R_{t-1}=i | x_{0:t-1} ), i \notin S_{AR}
            
            %predi(t,:,map_from_2{:}); % P( N_{t}=n, R_{t}=j | x_{0:t-1} ), j \in S_{AR}
            % = \sum_{k \notin S_{AR}} p_kj P( N_{t-1}=n-1, R_{t-1}=k | x_{0:t-1} ),
            
            %si(t,:,map_from_2{:}); % P( N_{t}=n, R_{t}=r | x_{0:T} ), j \in S_{AR}
            
            for i = 1:L-l1
                for r = 1:L
                    % temp = P( R_t = r, N_t = n (>2) | x_{0:T} ) * P( R_{t-1} = i, N_{t-1} = n-1 | x_{0:t-1} ) / P( R_t = r, N_t = n (>2) | x_{0:t-1})
                    temp = si(t,r,map_from_2{:}).*fi(t-1,i+l1,map_from_1{:})./predi(t,r,map_from_2{:}); % from a lemma in the thesis
                    temp(isnan(temp)|isinf(temp)) = 0; % just make sure none of the elements of temp are nans
                    s(i,r) = s(i,r) + sum(temp(:)); % sum over all N_t \in S^(t), and over t=1,...,T
                end
            end
        end
        
        % the work of Hamilton 1990 gives updates for p_ij
        for i = l1+1:L
            temp = si(2:T,i,:);
            P_hat(i,:) = s(i-l1,:).*P(i,:)./sum(temp(:));
            P_hat(i,:) = P_hat(i,:)./sum(P_hat(i,:));
        end
        P = P_hat; Pt = P';
        theta( end-(L^2-1):end ) = Pt(:);
        
        %% AR updates
        si_ar = zeros(T,trunc,l1);
        si_iid = zeros(T,L-l1);
        for t = 1:T
            for r = 1:l1
                rm = [repmat({':'},1,r-1),1,repmat({':'},1,l1-r)];
                for m = 1:trunc
                    temp_m = si(t,r,rm{:});
                    si_ar(t,m,r) = sum(temp_m(:)); % this contains the marginal smoothed probabilties P( R_t = r, N_{t,r} = m | x_{0:T} )
                    % for t = 1,...,T and for all r in S_{AR}
                    % and are calcualted by summing over all other counters
                    rm{r} = m+1;
                end
            end
            for r = l1+1:L
                temp_r = si(t,r,:);
                si_iid(t,r-l1) = sum(temp_r(:)); % this contains the marginal smoothed probabilties P( R_t = r | x_{0:T} ) for t = 1,...,T and r NOT in S_{AR}
                % and are calcualted by simply summing over all counters
            end
        end
        
        for r = 1:l1
            params = theta((1:3)+(r-1)*3);
            switch model{1}{r}
                case 'AR(1) type II'
                    f = @(phi) -Q(phi,r,data,si_ar,trunc); % Q is a 1D function of phi, and we need to maximise Q
                    phi_hat = fmincon(f,params(2),[],[],[],[],-1,1,[],options);
                    [~,alpha_hat,sigma_hat_squared] = Q(phi_hat,r,data,si_ar,trunc); % Q also gives updates for alpha and sigma, when phi is a maximiser
                    theta((1:3)+(r-1)*3) = [alpha_hat,phi_hat,sigma_hat_squared];
                case 'AR(1) type III'
                    [phi_hat, alpha_hat, sigma_hat_squared] = type_III_updates(data,si_ar(:,:,r),trunc);
                    theta((1:3)+(r-1)*3) = [alpha_hat,phi_hat,sigma_hat_squared];
            end
        end
        k=0;
        for r = 1:L-l1
            switch model{2}{r}
                case 'G'
                    sr = sum(si_iid(:,r));
                    theta(2+(r+l1-1)*3) = sum(si_iid(:,r).*data)./sr; % lemma in thesis
                    theta(3+(r+l1-1)*3) = sum(si_iid(:,r).*((data-theta(2+(r+l1-1)*3)).^2))./sr; % lemma in thesis
                case 'LN'
                    sr = sum(si_iid(:,r));
                    theta(2+(r+l1-1)*3) = sum(si_iid(:,r).*log(data))./sr; % lemma in thesis
                    theta(3+(r+l1-1)*3) = sum(si_iid(:,r).*((log(data)-theta(2+(r+l1-1)*3)).^2))./sr; % lemma in thesis
                case 'LN shifted'
                    %f = @(q) -Q_LN_shifted(q,data,si_iid(:,r),theta((1:3)+(r+l1-1)*3));    % Q is a 1D function of q, and we need to maximise Q
                    %queue = MIN:1:MAX; next=find(sort(data)>theta((1)+(r+l1-1)*3),1); for kk = 1:length(queue), F(kk) = (f(queue(kk))); end; plot(queue,F), hold on, Y = sort(data); f(Y(next)-eps*100),f(Y(next)-0.001), plot(Y(next)-eps*100,f(Y(next)-eps*100),'ro',Y(next)-0.001,f(Y(next)-0.001),'ko'), hold off, theta, drawnow, pause
                    %theta(1+(r+l1-1)*3), theta(1+(r+l1-1)*3)-data,f(theta(1+(r+l1-1)*3))
                    %try
                    %    next = find(Y>=theta((1)+(r+l1-1)*3),1);
                    %    MAX = Y(next)-0.01;
                    %    q_hat = fmincon(f,theta(1+(r+l1-1)*3),[],[],[],[],MIN-1,MAX,[],options);
                    %catch ME
                    %    Err = ME;
                    %    b = 1;
                    %    q_hat = theta(1+(r+l1-1)*3);
                    %end
                    %[~,mu_hat,sigma_hat_squared] = Q_LN_shifted(q_hat,data,si_iid(:,r),theta((1:3)+(r+l1-1)*3));
                    %theta((1:3)+(r+l1-1)*3) = [q_hat, mu_hat, sigma_hat_squared]; % lemma in thesis
                    [~,mu_hat,sigma_hat_squared] = Q_LN_shifted(theta(1+(r+l1-1)*3),data,si_iid(:,r),theta((1:3)+(r+l1-1)*3));
                    theta((1:3)+(r+l1-1)*3) = [theta(1+(r+l1-1)*3), mu_hat, sigma_hat_squared]; % lemma in thesis
                case 'LN shifted and reversed'
                    [~,mu_hat,sigma_hat_squared] = Q_LN_shifted_reversed(theta(1+(r+l1-1)*3),data,si_iid(:,r),theta((1:3)+(r+l1-1)*3));
                    theta((1:3)+(r+l1-1)*3) = [theta(1+(r+l1-1)*3), mu_hat, sigma_hat_squared]; % lemma in thesis
                case 'Gamma'
                    q3 = theta(1+(r+l1-1)*3);
                    dat = data(data>q3)-q3;
                    theta_fun = @(a) mean( dat ) / a;
                    log_gam_pdf = @(a) -sum(log(gampdf(  dat ,a,theta_fun(a) )));
                    a = fmincon(log_gam_pdf,theta(2+(r+l1-1)*3),[],[],[],[],0,Inf,[],options );
                    theta((1:3)+(r+l1-1)*3) = [q3,a,theta_fun(a)];
                case 'E'
                    next = find(Y>=theta((1)+(r+l1-1)*3),1);
                    q_hat = Y(next);
                    mu = sum( si_iid(:,r).*(data-q_hat) ) /sum(si_iid(:,r));
                    theta((1:2)+(r+l1-1)*3) = [q_hat,mu];
                case 'Beta'
            end
        end
        
        tol = abs(theta_last_iteration - theta);
        ll_tol = abs(likelihood-ll_old);
        ll_old = likelihood;
        theta_last_iteration = theta;

        if any(theta(3:3:end-L^2) < eps)
            fprintf('a variance term has been sent to 0')
            break
        end
        if any(theta(end-L^2+1:end) < eps)
            fprintf('a p_ij term has been sent to 0')
            break
        end
        
        if display==1
            theta
            fprintf('iterations = %i, tol = %e \n',[its_count,max(tol)])
        end
    end
end

[fi,predi,likelihood] = forward(data, theta, p0, model,trunc,bw);


MLEs = theta;
end






function [fi,predi,loglikelihood] = forward(data, params, p0, model, trunc, bw)
%forward calculates the smoothed inferences for the MRS model, with initial
%distribution on the hidden chain of p0, and parameters theta
% Inputs
%   - data, a column vector of observed data
%   - theta, a vector of parameters
%   - p0, the initial distribution on the hidden chain R_t
%   - model, the model specification
% Outputs
%   - fi, the filtered inference,
%     P(H_t = (N_t,R_t)|x_{0:t})

l1 = length(model{1});
l2 = length(model{2});
L = l1+l2;
T = length(data);
P = reshape(params( (end-(L^2-1)):end ),L,L)';

sz_F_ar = [l1,T,trunc];
F_ar=zeros(sz_F_ar);
sz_F_iid = [l2,T];
F_iid = zeros(sz_F_iid);

for r = 1:l1
    % find the model pdf and evaluate it for each t and each lag
    switch model{1}{r}
        case 'AR(1) type II'
            F_ar(r,:,:) = ARpdf_vectorised(params((1:3)+(r-1)*3),2, data, trunc);
        case 'AR(1) type III'
            F_ar(r,:,:) = ARpdf_vectorised(params((1:3)+(r-1)*3),3, data, trunc);
    end
end

for r = l1+1:L
    % find the model pdf
    switch model{2}{r-l1}
        case 'G'
            F_iid(r-l1,:) = 1/sqrt(2*pi*params(3+(r-1)*3)) .* exp(-(data-params(2+(r-1)*3)).^2/(2*params(3+(r-1)*3))); % normpdf(data,params(2+(r-1)*3),sqrt(params(3+(r-1)*3)));
            F_iid(r-l1,F_iid(r-l1,:)==0) = realmin;
        case 'LN'
            dat = data;
            idx = dat>0;
            dat = dat(idx);
            F_iid(r-l1,idx) = (1./(dat*sqrt(2*pi*params(3+(r-1)*3)))) .* exp(-( log(dat)-params(2+(r-1)*3) ).^2/(2*params(3+(r-1)*3))); % lognpdf(data,params(2+(r-1)*3),params(3+(r-1)*3));
        case 'LN shifted'
            dat = data-params(1+(r-1)*3);
            idx = dat>0; 
            dat = dat(idx);
            F_iid(r-l1,idx) = (1./(dat*sqrt(2*pi*params(3+(r-1)*3)))) .* exp(-( log(dat)-params(2+(r-1)*3) ).^2/(2*params(3+(r-1)*3)));
            % lognpdf(dat,params(2+(r-1)*3),params(3+(r-1)*3)),
        case 'LN shifted and reversed'
            dat = data-params(1+(r-1)*3);
            idx = dat<0;
            dat = -dat(idx);
            F_iid(r-l1,idx) = lognpdf(dat,params(2+(r-1)*3),params(3+(r-1)*3));
        case 'Gamma'
            dat = data-params(1+(r-1)*3);
            idx = dat>0; 
            dat = dat(idx);
            F_iid(r-l1,idx) = gampdf(dat,params(2+(r-1)*3),params(3+(r-1)*3));
        case 'E'
            F_iid(r-l1,:) = exppdf(data-params(1+(r-1)*3),params(2+(r-1)*3));
        case 'Beta'
            F_iid(r-l1,:) = betapdf((data-params(1+(r-1)*3))/( 300-params(1+(r-1)*3) ), params(2+(r-1)*3) , params(3+(r-1)*3));
    end
    % evaluate it
end
%permute(F_ar,[2,3,1]), F_iid,
sz = [L,1*ones(1,l1)];

a_0 = p0(:);
a_tilde_0 = a_0(:).*[F_ar(:,1,1);F_iid(:,1)];
c_0 = sum(a_tilde_0(:));
a_tilde_0 = a_tilde_0/c_0;

a_hat_t = zeros(sz);
N = ones(1,l1);
for r=1:L
    a_hat_t(MapTo(sz,[r,N])) = a_tilde_0(r);
end

SZ = [T,L,trunc*ones(1,l1)];
if bw==0
    fi = zeros(SZ); % 1st index is t, 2nd index is regime, elements
    % are cell arrays. The n=(n1,n2,...,nl1)th entry is
    % P(R_t = i, N_t = n | x_0,...,x_t ),
    predi = zeros(SZ); % 1st index is t, 2nd index is regime, elements
    % are cell arrays. The n=(n1,n2,...,nl1)th entry is
    % P(R_t = i, N_t = n | x_0,...,x_t ),
else
    fi = [];
    predi = [];
end
dots  = repmat({':'},1,l1);

loglikelihood = log(c_0);

rm = repmat({},1,l1);
rm_trunc = repmat({},1,l1);
fromidx = cell(l1,l1+1);
for j = 1:l1
    fromidx(j,:) = [{j} dots];
end
from_idx = [{l1+1:L} dots];
from_idx_mt = [{l1+1:L} repmat({1:trunc-1},1,l1)];
saveindex = [{1} {':'} repmat({1:1},1,l1)];
fi(saveindex{:}) = a_hat_t;

sz_all = [L*ones(T-1,1),[repmat((2:trunc)',1,l1);repmat(trunc,T-trunc,l1)]];

for t = 2:T
    %%
    sz = sz_all(t-1,:);
    
    a_t = zeros(sz);
    temp_a_t = zeros(sz+[0,ones(1,l1)]);
    a_tilde_t = zeros(sz);
    
    mt = min(t,trunc);
    t_range = 2:mt;
    t_range_trunc = 2:mt+1;
    for i = 1:l1
        rm{i} = t_range;
        rm_trunc{i} = t_range_trunc;
    end
    
    for j = 1:l1
        if t > trunc
            rm_j = rm_trunc; rm_j{j} = 1;
            temp = a_hat_t(j,dots{:});
            for i = 1:L
                temp_a_t(i,rm_j{:}) = sum(temp,j+1).*P(j,i); % P( R_t = r, N_t = n  | x_0,...,x_{t-1} ), where N(j) = 1
            end
        else
            rm_j = rm; rm_j{j} = 1;
            temp = a_hat_t(j,dots{:});
            for i = 1:L
                a_t(i,rm_j{:}) = sum(temp,j+1).*P(j,i); % P( R_t = r, N_t = n  | x_0,...,x_{t-1} ), where N(j) = 1
            end
        end
    end
    if t > trunc
        temp = a_hat_t(from_idx{:});
        for i = 1:L
            temp_i = temp.*repmat(P(l1+1:L,i),[1,repmat(mt,1,l1)]);
            temp_a_t(i,rm_trunc{:}) = sum(temp_i,1); % P( R_t = r, N_t = n  | x_0,...,x_{t-1} ), where N(j) > 1
        end
    else
        temp = a_hat_t(from_idx{:});
        for i = 1:L
            temp_i = temp.*repmat(P(l1+1:L,i),[1,repmat(mt-1,1,l1)]);
            a_t(i,rm{:}) = sum(temp_i,1); % P( R_t = r, N_t = n  | x_0,...,x_{t-1} ), where N(j) > 1
        end
    end
    if t > trunc
        if l1==2
            temp_a_t(:,1:end-1,end-1) = temp_a_t(:,1:end-1,end-1) + temp_a_t(:,1:end-1,end);
            temp_a_t(:,end-1,1:end-1) = temp_a_t(:,end-1,1:end-1) + temp_a_t(:,end,1:end-1);
            temp_a_t(:,end-1,end-1) = temp_a_t(:,end-1,end-1) + temp_a_t(:,end,end);
            a_t(:,dots{:}) = temp_a_t(:,1:end-1,1:end-1);
        elseif l1==1
            temp_a_t(:,end-1) = temp_a_t(:,end-1) + temp_a_t(:,end);
            a_t(:,dots{:}) = temp_a_t(:,1:end-1);
        end
    end
    
    
    for i = 1:l1
        order = repmat(repmat(1:mt,mt^(i-1),1),1,mt^(l1-i));
        order = order(:);
        densities = F_ar(i,t,order); densities=densities(:)';
        a_tilde_t(i,:) = a_t(i,:).*densities;
    end
    
    a_tilde_t(l1+1:L,:) = a_t(l1+1:L,:).*F_iid(1:l2,t); % f(x_t, R_t = r, N_{t} = n | x_0,...,x_{t-1} ),
    
    c_t = sum(a_tilde_t(:)); % f(x_t | x_0,...,x_{t-1} ),
    a_hat_t = a_tilde_t/c_t; % f( R_t = r, N_t = n | x_0,...,x_t ),
    
    if bw==0
        t_range = 1:mt;
        saveindex{1} = t;
        for i = 3:l1+2
            saveindex{i} = t_range;
        end
        fi(saveindex{:}) = a_hat_t;
        predi(saveindex{:}) = a_t;
    end
    
    loglikelihood = loglikelihood + log(c_t);
    
end

end

function [f] = ARpdf_vectorised(params, type, data, trunc)
%ARpdf_typeII calcualates the conditional pdf of an AR(1) process given the
%observation at some lag for MRS model
% Inputs
%   - data, a scalar value, x_t
%   - params, paramters [\alpha,\phi,\sigma]
%   - lag, the number of time steps before t when lagged_data was observed
%   - x_{t-lag}
%   - type, either 2 or 3 depending on whether the MRS model is type 1 or 2
% Outputs
%   - f, the value of the conditional pdf
T = length(data);
data = data(:);

f = zeros(T,trunc);

LD = zeros(T,trunc);
for tea=2:trunc
    LD(tea:end,tea-1) = data(1:T-tea+1); % a lower triangular matrix of lagged values
end

if type == 2
    indices = tril(ones(T,trunc),-1); % a lower triangular matrix of 1s
    B = params(2).^([1:trunc-1,Inf]).*indices; % phi^m
    C = tril(((1-B)./(1-params(2))),0); % (1-phi^m)/(1-phi)
    mu = params(1)*C+B.*LD; % alpha * (1-phi^m)/(1-phi) + phi^m x_{t-m}
    C2 = tril(((1-B.^2)./(1-params(2)^2)),0); % (1-phi^2m)/(1-phi^2)
    s = tril(C2*params(3),0); s(s==0)=1; % sigma^2 (1-phi^2m)/(1-phi^2)
    f = tril(exp(-((data-mu).^2)./(2*s))./sqrt(2*s*pi),0); % the normal density,
elseif type == 3
    temp_f = tril((data-params(1)-params(2)*LD).^2,-1);
    temp_f = tril(exp( -(temp_f)./(2*params(3)) ),-1);
    f = temp_f./sqrt(2*params(3)*pi); % the normal density,
    f(eye(T,trunc)==1) = 1; %f(eye(T)==1) = exp( -((data-params(1)/(1-params(2))).^2)./(2*params(3)/(1-params(2))) ) ./ sqrt(2*params(3)*pi/(1-params(2)));
    f(:,end) = exp( -(data-params(1)/(1-params(2))).^2/(2*params(3)/(1-params(2)^2)) )/sqrt(2*pi*params(3)/(1-params(2)^2));
end
indices = tril(ones(T,trunc),-1);
f(f==0 & indices==1) = realmin;

end

function [to] = MapTo(sz,H)
%MapTo maps th vector H = (R_t,N) to a position in an array of size sz
cp = [1,cumprod(sz(1:end-1))]';
to = (H-1)*cp + 1;
end

function [si] = backward(si,predi,P,l1,T,trunc)
%BACKWARD implements the backward to calculate the smoothed inferences 'si'
% using the filtered inferences 'fi' and the transition martix 'P'.
dots  = repmat({':'},1,l1);
L = size(P,1);
rmP = zeros([L,L,repmat(trunc,1,l1)]);

rm_mt_from = repmat({[2:trunc,trunc]},1,l1);
rm_mt_to = repmat({1:trunc},1,l1);
for r = 1:L
    rmP(r,:,dots{:}) = permute(repmat(P(r,:)',[1,repmat(trunc,1,l1)]),[l1+2,1:l1+1]);
end
for t = T-1:-1:1
    if t > trunc-1
        map_from = rm_mt_from;
        map_to = rm_mt_to;
    else
        map_from = repmat({2:t+1},1,l1);
        map_to = repmat({1:t},1,l1);
    end
    for r = l1+1:L % r \notin S_AR
        ratio_t = si(t+1,:,map_from{:})./predi(t+1,:,map_from{:}).*rmP(r,:,map_from{:});
        ratio_t(isnan(ratio_t)) = 0;
        S = sum(ratio_t,2);
        si(t,r,map_to{:}) = si(t,r,map_to{:}).*S;
    end
    for r = 1:l1 % r \notin S_AR
        rm = map_from;
        rm{r} = 1;
        ratio_t = si(t+1,:,rm{:})./predi(t+1,:,rm{:}).*rmP(r,:,rm{:});
        ratio_t(isnan(ratio_t)) = 0;
        S = sum(ratio_t,2);
        si(t,r,map_to{:}) = si(t,r,map_to{:}).*S;
    end
    
end

end

function [Q,alpha_hat,sigma_hat_squared] = Q(phi,r,data,si_ar,trunc)
%ARpdf_typeII calcualates the conditional pdf of an AR(1) process given the
%observation at some lag for MRS model
% Inputs
%   - data, a vecotr value, x_t
%   - phi,
%   - type, either 2 or 3 depending on whether the MRS model is type 1 or 2
% Outputs
%   - Q as a function of phi
%   - updates for alpha, and sigma

T = length(data);
data = data(:);

LD = zeros(T,trunc);
for n=2:trunc
    LD(n:end,n-1) = data(1:T-n+1);
end
LD(eye(T,trunc)==1) = 0;

indices = tril(ones(T,trunc),-1);
P = phi.^([1:trunc-1,inf]).*indices;

B1 = tril((1-P)./(1-phi),0);
B2 = tril((1-P.^2)./(1-phi^2),0);

A_tm = tril((B1).*(1+phi)./(1+B1),0);
temp_Q = tril((data - P.*LD),0);
B_tm = tril(temp_Q.*(1+phi)./(1+B2),0);

alpha_hat = sum(sum(si_ar(:,:,r).*B_tm))./sum(sum(si_ar(:,:,r).*A_tm));

temp_Q = tril((temp_Q - alpha_hat.*B1).^2,0);
C_tm = tril(temp_Q./B2,0);

sigma_hat_squared = sum(sum(si_ar(:,:,r).*C_tm))./sum(sum(si_ar(:,:,r)));

L_tm = tril(-0.5*log(B2) - 0.5*log(sigma_hat_squared),0);% - 0.5*temp_Q./(sigma_hat_squared.*B2);

Q = sum(sum(si_ar(:,:,r).*L_tm));



end

function [phi_hat, alpha_hat, sigma_hat_squared] = type_III_updates(data,si_ar,trunc)
%TYPE_III_UPDATES does what it was on the box; constructs the parameter
%updates for type III AR(1) process
%   Inputs,
%       - the data, a vector
%       - si_ar, an array of the smoothed probabilites, first index is tm
%       second index is lag, m
%   Outputs
%       - the miximisers of Q

data = data(:);
T = length(data);
si_ar(eye(T,trunc)==1) = 0;
si_ar(:,end) = 0;
LD = zeros(T,trunc);
for n=2:trunc
    LD(n:end,n-1) = data(1:T-n+1);
end
LD(eye(T,trunc)==1) = 0;

A_1 = sum(sum(si_ar.*LD));
A_2 = sum(sum(si_ar.*(LD.^2)));

S = sum(sum(si_ar));

alpha_hat = sum(sum( si_ar.*data.* (A_2-A_1*LD) )) / (A_2*S-A_1^2); %sum(sum(si_ar.*data.*(A_2-A_1*LD))) / (A_2 - A_1^2);

phi_hat = (sum(sum(si_ar.*(data.*LD))) - alpha_hat*A_1)/A_2;

sigma_hat_squared = sum(sum(si_ar.* ((data - alpha_hat - phi_hat*LD).^2) )) / S;

end

function [Q,mu_hat,sigma_hat_squared] = Q_LN_shifted(q,data,si_iid,old_params)
%Constructs the function Q for a shifted lognormal distribution, we can
% then optimise it!
% Inputs
%       - q, a scalar
%       - the data, a column vector
%       - si_iid, the smoothed inferences, first index is time,
% Outputs
%       - a function value Q
%       - the maximisers mu_hat and sigma_hat_squared
T=length(data);
data = data(:);
data = data-q;

index = data>0;
if any(index==0&si_iid>0)
    Q = -Inf;
    mu_hat = old_params(2);
    sigma_hat_squared = old_params(3);
else
    data = data(index);
    ldata = log(data);
    
    si_iid = si_iid(index);
    
    sr = sum(si_iid);
    
    mu_hat = sum(si_iid.*ldata)/sr;
    sigma_hat_squared = sum( si_iid.* ((ldata-mu_hat).^2) )/sr;
    
    Q = -sum(si_iid.*ldata)-0.5*sr*log(sigma_hat_squared);
end


end


function [Q,mu_hat,sigma_hat_squared] = Q_LN_shifted_reversed(q,data,si_iid,old_params)
%Constructs the function Q for a shifted lognormal distribution, we can
% then optimise it!
% Inputs
%       - q, a scalar
%       - the data, a column vector
%       - si_iid, the smoothed inferences, first index is time,
% Outputs
%       - a function value Q
%       - the maximisers mu_hat and sigma_hat_squared
T=length(data);
data = data(:);
data = q-data;

index = data>0;
if any(index==0&si_iid>0)
    Q = -Inf;
    mu_hat = old_params(2);
    sigma_hat_squared = old_params(3);
else
    data = data(index);
    ldata = log(data);
    
    si_iid = si_iid(index);
    
    sr = sum(si_iid);
    
    mu_hat = sum(si_iid.*ldata)/sr;
    sigma_hat_squared = sum( si_iid.* ((ldata-mu_hat).^2) )/sr;
    
    Q = -sum(si_iid.*ldata)-0.5*sr*log(sigma_hat_squared);
end


end

