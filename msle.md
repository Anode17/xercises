Maximum Simulated Likelihood
================
Aslan Bakirov

Sometimes limited dependent variable (LDV) models are so heterogenous
that the probability value varies individually. In those cases, it would
require us integrating over the unobserved heterogeneity, which is
analytically an impasse, but quite manageable with simulation:

$$f(y_i|x_i)=E_u[~f(y_i|x_i,u_i)~]=\int_uf(y_i|x_i,u_i)~f(u)du$$ The
expression above shows that **CEF** of y given x can be expressed as an
expectation over the support of *u*, an unobserved variation in our CEF.
Now having the new CEF, to derive inference on that we need to integrate
it, over the support of *u*, yet we can’t because the distribution/ mass
of this variation is unobserved.  
To proceed, we should solve the integral with simulation methods. As
such, in the following steps we can handle it:

-   assume $u_i \sim N(0,\sigma_u)$  
-   draw a random subsample from the above distribution, size S, $u^S$  
-   for an individual $i$:
    -   estimate the subsimulator $\tilde f(y_i|x_i,u^s_i)$ with first
        element of the above draw
        -   repeat the last step for all elements of the *u^s*
    -   average them out: $\frac{1}{S}\sum_s \tilde f(y_i|x_i,u^s_i)$  
-   repeat for all individuals $i$ in the sample  
-   $logLik=\sum_i^N log(\tilde f(y_i|x_i,u^s_i))$ is the simulated
    likelihood to be maximized.

## Random Logit example

This model works with a probability of choosing Y given covariate X,
just like in a basic logit, with an exception:
$$P[y_i=1|x_i]=\Lambda(\beta_1+\beta_{2i}x_i)$$ that is, we would have
to estimate *N+1* parameters, $\beta_2$ for each individual and a
$\beta_1$. But we can proceed by modeling this variation as unobserved
heterogeneity in $\beta_2$, such as:  
$$\beta_{2i}=\beta_2+u_i\sim N(0,\sigma_u)$$
$$u_i=\frac{\beta_{2i}-\beta_2}{\sigma_u}\sim N(0,1)$$

Expressing it as the integral above we get:

$$E_u[~f(y_i|x_i,u_i)~]=\int_u\Lambda(\beta_1+(\beta_{2}+\sigma_u)x_i)^{y_i}*\{1-\Lambda(\beta_1+(\beta_{2}+\sigma_u)x_i)\}^{1-y_i}~f(u_i)du_i$$
Which doesn’t have a closed-form solution. Need to simulate.(

``` r
## import the data
Odds <- read_excel("Odds.xlsx") %>% data.frame() %>% select(-...3) %>% filter(Year>2007) 

## run the initial logit to see the coefs

df<-data.frame(Y=Odds$Best.Picture.Oscar.statuette., X=Odds$Metacritic,sllik=as.numeric(length(Odds)))

lgt1<-glm(Y ~ X,data=df,family = binomial())

## saving the coefficients
beta_1<-as.numeric(lgt1$coefficients[1])
beta_2<-as.numeric(lgt1$coefficients[2])

## and white sd errors
sds<-coeftest(lgt1, vcov.=vcovHC(lgt1, type="HC1"))[2,2]


## assuming Normal

set.seed(1701)

N=1000

u_s= rnorm(N,0,1)

## loglikelihood function:  
 llik<- function(param) {
    #p_win=1/(1+exp(-1*(b1+(b2+u_s*sd2)*df$X[i])))

   b1<- param[1]
   b2<- param[2]
   sd2<- param[3]
   
  ll=mean(
    sapply(
      c(df$Y,df$X),
      function(Y,X) log(1/(1+exp(-1*(b1+(b2+u_s*sd2)*df$X))))*df$Y+(1-df$Y)*log(1-1/(1+exp(-1*(b1+(b2+u_s*sd2)*df$X))))
      )
    )
  return(ll)
}

## optimizing the log-likelihood:  
 
res<-maxLik(llik,start = c(b1=beta_1,b2=beta_2,sd2=sds))

round(res$estimate,5)
```

    ##        b1        b2       sd2 
    ## -16.39510   0.16520   0.00005

``` r
knitr::kable(res$hessian)
```

|     |         b1 |          b2 |          sd2 |
|:----|-----------:|------------:|-------------:|
| b1  | -0.0889844 |   -7.976675 |   -0.4444223 |
| b2  | -7.9766749 | -717.978843 |  -39.3822197 |
| sd2 | -0.4444223 |  -39.382220 | -719.5797291 |
