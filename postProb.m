function posterior = postProb(X,y,xtest,target)
  %get the posterior from xtest and the target class
  
  likelihoods = [];
  for i=1:length(xtest(1,:))
    likelihoods = [likelihoods; getLikelihood(X(:,i),y,xtest(i),target)];
  end
  
  posterior = log10(prod(likelihoods)*getPrior(y,target));
end