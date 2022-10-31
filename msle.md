Maximum Simulated Likelihood
================
Aslan Bakirov

Sometimes limited dependent variable (LDV) models are so heterogenous
that the probability value varies individually. In those cases, it would
require us integrating over the unobserved heterogeneity, which is
analytically an impasse, but quite manageable with simulation:

$$f(y_i|x_i)= E_u[~f(y_i|x_i,u_i)~] = \int_uf(y_i|x_i,u_i)~f(u)du$$

The expression above shows that **CEF** of y given x can be formulated
as an expectation over the support of *u*, an unobserved variation in
our CEF. Now having the new CEF, to derive inference on that we need to
integrate it, over the support of *u*, yet we can’t because the
distribution/ mass of this variation is unobserved.  
To proceed, we should solve the integral with simulation methods. As
such, in the following steps we can handle it:

-   assume a distribution/mass, e.g. $u_i \sim N(0,\sigma)$  
-   draw a random subsample from the above distribution, size S, $u^S$  
-   for an individual $i$:
    -   estimate the subsimulator $\tilde f(y_i|x_i,u^s_i)$ with first
        element of the above draw
        -   repeat the last step for all elements of the *u^s*
    -   average them out: $\frac{1}{S}\sum_s \tilde f(y_i|x_i,u^s_i)$  
-   repeat for all individuals $i$ in the sample  
-   The simulated likelihood to be maximized becomes:

$$ logLik = \sum_i ^N log( \tilde f( y_i | x_i, u ^s_i) ) $$

## Random Logit example

This model works with a probability of choosing Y given covariate X,
just like in a basic logit, with an exception:
$$P[y_i=1|x_i]=\Lambda(\beta_1+\beta_{2i}x_i)$$ that is, we would have
to estimate *N+1* parameters, $\beta_2$ for each individual and a
$\beta_1$. But we can proceed by modeling this variation as unobserved
heterogeneity in $\beta_2$, such as:  
$$\beta_{2i}=\beta_2+u_i\sim N(0,\sigma)$$
$$u_i=\frac{\beta_{2i}-\beta_2}{\sigma}\sim N(0,1)$$

Expressing it as the integral above we get:

$$ \int_u\Lambda(\beta_1+(\beta_{2}+\sigma ~u_i)x_i)^{y_i}* 
\\{1-\Lambda(\beta_1+(\beta_{2}+\sigma ~u_i)x_i)\\}^{1-y_i}~f(u_i)du_i $$

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

N=500

u_s= rnorm(N,0,1)

## loglikelihood function:  
 llik<- function(param) {
   b1<- param[1]
   b2<- param[2]
   sd2<- param[3]
   
  ll=mean(
    mapply(function(X,Y) mean(log(1/(1+exp(-1*(b1+(b2+u_s*sd2)*X))))*Y+(1-Y)*log(1-1/(1+exp(-1*(b1+(b2+u_s*sd2)*X))))),df$X,df$Y))
  
  return(ll)
}

## optimizing the log-likelihood:  
 
res<-maxLik(llik,start = c(b1=0,b2=0,sd2=0))

## resulting estimates
knitr::kable(rbind(simulated=round(res$estimate,5),originals=c(beta_1,beta_2,sds)),row.names = TRUE)
```

|           |        b1 |        b2 |       sd2 |
|:----------|----------:|----------:|----------:|
| simulated | -16.46879 | 0.1662000 | 0.0000000 |
| originals | -16.46906 | 0.1662065 | 0.0465801 |

``` r
## hessian
knitr::kable(res$hessian)
```

|     |         b1 |          b2 |          sd2 |
|:----|-----------:|------------:|-------------:|
| b1  | -0.0890399 |   -7.972623 |   -0.6986633 |
| b2  | -7.9726226 | -717.128246 |  -62.8452845 |
| sd2 | -0.6986633 |  -62.845284 | -665.6435403 |

## Experiment with simulated data

``` r
set.seed(1701)

N=500

u_s<-rnorm(N)

df<-data.frame(X=rnorm(N, 1, 2))
 
df$Y<-rbinom(N, 1, p=1/(1+exp(-1*(-1+(2.3+u_s*0.4)*df$X))))

## checking the balance
table(df$Y)
```

    ## 
    ##   0   1 
    ## 181 319

``` r
## the likelihood function

 llik<- function(param) {
  
  b1<- param[1]
  b2<- param[2]
  sd2<- param[3]
   
  ll=mean(
    mapply(function(X,Y) mean(log(1/(1+exp(-1*(b1+(b2+u_s*sd2)*X))))*Y+
                                (1-Y)*log(1-1/(1+exp(-1*(b1+(b2+u_s*sd2)*X))))), df$X, df$Y))
  return(ll)
 }
 
## running the optimization
res<-maxLik(llik,start = c(b1=-0.5,b2=2,sd2=0.02))

## resulting estimates
knitr::kable(rbind(simulated=round(res$estimate,5),originals=c(-1,2.3,0.4)),row.names = TRUE)
```

|           |       b1 |      b2 | sd2 |
|:----------|---------:|--------:|----:|
| simulated | -0.79733 | 2.14461 | 0.0 |
| originals | -1.00000 | 2.30000 | 0.4 |

``` r
## and the hessian
knitr::kable(round(res$hessian,4))
```

|     |      b1 |      b2 |     sd2 |
|:----|--------:|--------:|--------:|
| b1  | -0.0726 | -0.0382 | -0.0033 |
| b2  | -0.0382 | -0.0659 | -0.0058 |
| sd2 | -0.0033 | -0.0058 | -0.0611 |
