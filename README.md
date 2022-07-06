``` r
library(pacman)
pacman::p_load(quantmod, ggplot2, forecast, tseries, rugarch, prophet, tsfknn, fmxdat, tidyverse,tbl2xts)
```

``` r
JSE_TOP40 <-  fmxdat::SA_Indexes |> filter(ShareName=="JSE Top 40 Index Total Return Value") #daily (excl weekends) data  2002-06-21 to 2020-07-31. NB, how to handle the empty weekend spots??

JSE_Top40_xts <- tbl_xts(JSE_TOP40)
```

``` r
fmxdat::SA_Indexes |> group_by(Tickers, ShareName) |> summarise() #use to see which indexes to include. Current idea is to compare the  
```

    ## `summarise()` has grouped output by 'Tickers'. You can override using the
    ## `.groups` argument.

    ## # A tibble: 18 x 2
    ## # Groups:   Tickers [18]
    ##    Tickers        ShareName                                                     
    ##    <chr>          <chr>                                                         
    ##  1 FINI15TR Index FTSE/JSE Africa Financials 15 Index Total Return Value        
    ##  2 INDI25TR Index FTSE/JSE Africa Industrials 25 Index Total Return Value       
    ##  3 J205LCTR Index FTSE/JSE Large Cap Total Return Index                         
    ##  4 J430TR Index   FTSE/JSE Cap SWIXTop40 Total Return Value                     
    ##  5 J433TR Index   FTSE/JSE Africa SWIX All Share Index Capped Total Return Value
    ##  6 JALSHTR Index  FTSE/JSE Africa All Share Index Total Return Value            
    ##  7 JCAP40TR Index JSE Capped Top 40 Index Total Return Value                    
    ##  8 JGROWTR Index  FTSE/JSE All Share Style Growth Total Return Index            
    ##  9 JMOTETR Index  FTSE/JSE Africa Mobile Telecommunications Total Return Index  
    ## 10 JNCCGTR Index  FTSE/JSE Africa Health Care Total Return Index                
    ## 11 JSAPYTR Index  FTSE/JSE Africa Property Index Total Return Value             
    ## 12 JSHR40TR Index JSE Shareholder Weighted Top 40 Index Total Return Value      
    ## 13 JSHRALTR Index FTSE/JSE Africa SWIX All Share Index Total Return Value       
    ## 14 JSMLCTR Index  FTSE/JSE Africa Small Cap Index                               
    ## 15 JVALUTR Index  FTSE/JSE All Share Style Value Total Return Index             
    ## 16 MIDCAPTR Index JSE Mid Cap Index Total Return                                
    ## 17 RESI20TR Index FTSE/JSE Africa Resource 10 Index Total Return Value          
    ## 18 TOP40TR Index  JSE Top 40 Index Total Return Value

``` r
chartSeries(JSE_Top40_xts,TA=c(addMACD()))
```

![](README_files/figure-markdown_github/Visualise%20the%20Series-1.png)

# Forecasting using the Prophet Algorithm

``` r
#Prophet Forecasting
#Use data frame format instead of xts data format

JSE_TOP40_df <-  data.frame(ds=index(JSE_Top40_xts), y=as.numeric(JSE_Top40_xts$Price))

prediction_prophet <- prophet(JSE_TOP40_df)
```

    ## Disabling daily seasonality. Run prophet with daily.seasonality=TRUE to override this.

``` r
future_df <-  make_future_dataframe(prediction_prophet,periods=500)

forecast_prophet <-  predict(prediction_prophet,future_df)
```

``` r
#Generating the data set based on trained predictions and compare the prophet predictions to the actual observations

predicted_df <-  data.frame(forecast_prophet$ds,forecast_prophet$yhat)

length_of_train <-  length(JSE_Top40_xts$Price)

predicted_train_df <-  predicted_df[c(1:length_of_train),]
```

``` r
#Visualizing train prediction vs real data
g <- ggplot()+
  geom_smooth(aes(x= predicted_train_df$forecast_prophet.ds , y= JSE_Top40_xts$Price),
              colour="blue", level=0.99, fill="#69b3a2", se=T) +
  geom_point(aes(x= predicted_train_df$forecast_prophet.ds ,y=predicted_train_df$forecast_prophet.yhat), size = 0.3, colour="black")+
    labs(x = "Date",
         y = "Total Index Return",
         color = "Legend") +     ## Notice that this graph can be improved apon severely by including legends
    scale_color_manual(values = colors)+
  ggtitle("Prophet: Training Prediction vs. Real Data")
g
```

    ## Don't know how to automatically pick scale for object of type xts/zoo. Defaulting to continuous.

    ## `geom_smooth()` using method = 'gam' and formula 'y ~ s(x, bs = "cs")'

![](README_files/figure-markdown_github/Visualise%20Train%20Prediction%20vs%20Observed%20Data-1.png)

``` r
#Here we investigate the accuracy of the predictions with cross validation

accuracy_of_forecast<- accuracy(predicted_train_df$forecast_prophet.yhat,JSE_TOP40_df$y)

detrended_forecasts <- prophet_plot_components(prediction_prophet,forecast_prophet) #To have a clearer understanding of the data generating process, I plot the forecasted prophet components divided by a trend component, weekly seasonality and yearly seasonality.
```

![](README_files/figure-markdown_github/Cross%20Validation-1.png)

# The K-Nearest Neighbors (KNN) Algorithm

``` r
# Remember here to justify the selection of the k-value, the lags, as well as the msas. # Do so by comparing the RMSE, MAE, and the MAPE values for different specifications. 

KNN_prediction <- knn_forecasting(JSE_TOP40_df$y, h = 500 , lags = 1:30, k = 30, msas = "MIMO" ) 

#'h is the number of values to forecast'. 'k' is the parameter in the KNN regression. 'lags' is the order of lags used in the AR process. 'msas' is a string indicating the Multiple-Step Ahead Strategy used when more than one value is predicted. It can be "recursive" or "MIMO" (the default). 

#Accuracy of the model's training  set 

rolling_origin <- rolling_origin(KNN_prediction)

print(rolling_origin$global_accu)  # Provides the RMSE, MAE and the MAPE
```

    ##      RMSE       MAE      MAPE 
    ## 662.68875 481.60708   6.94695

``` r
autoplot(KNN_prediction)
```

![](README_files/figure-markdown_github/KNN%20Prediction-1.png)

# The Feed-forward Neural Network (FNN)

“A feed-forward neural network (FNN) is an artificial neural network
wherein connections between the nodes do not form a cycle. As such, it
is different from its descendant: recurrent neural networks.

The feed-forward neural network was the first and simplest type of
artificial neural network devised. In this network, the information
moves in only one direction—forward—from the input nodes, through the
hidden nodes (if any) and to the output nodes. There are no cycles or
loops in the network ”

``` r
#Fitting  the nnetar

Lambda = BoxCox.lambda(JSE_Top40_xts$Price) 

# I select the specific number of hidden nodes is half of the number of input nodes (including external regressors, if given) plus 1.
# To ensure that the residuals will be approximately homoscedastic, a Box Cox lambda is approach is applied. I forecast the next 500 values with the neural net fitted. I then proceed to apply the nnetar function with the lambda assigned as parameters.

FNN_fit = nnetar(JSE_Top40_xts$Price,lambda=Lambda)

FNN_fit # See the output results 
```

    ## Series: JSE_Top40_xts$Price 
    ## Model:  NNAR(1,1) 
    ## Call:   nnetar(y = JSE_Top40_xts$Price, lambda = Lambda)
    ## 
    ## Average of 20 networks, each of which is
    ## a 1-1-1 network with 4 weights
    ## options were - linear output units 
    ## 
    ## sigma^2 estimated as 0.000548

``` r
# NB- avoid running code over and over as computing time is rather strenuous. 

FNN_forecast <-  forecast(FNN_fit,PI=T,h=500) # Forecast 500 periods (week days) ahead as in previous predictions.
  
autoplot(FNN_forecast) # A completely opposite result to KNN forecast. 
```

![](README_files/figure-markdown_github/Visualise%20FNN%20Forecast-1.png)

``` r
accuracy(FNN_fit) # Interpreting the forecasts. 
```

    ##                    ME     RMSE     MAE          MPE    MAPE     MASE       ACF1
    ## Training set 1.954182 55.17898 36.4821 0.0008798428 0.99617 1.047369 0.04608366
