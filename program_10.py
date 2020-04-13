#!/bin/env python
# Created on March 25, 2020 by Keith Cherkauer
# Revised by Alka Tiwari on April 11, 2020
# This script serves as the solution set for assignment-10 on descriptive
# statistics and environmental informatics.  See the assignment documention 
# and repository at:
# https://github.com/Environmental-Informatics/assignment-10.git for more
# details about the assignment.

# important libraries
import pandas as pd
import scipy.stats as stats
import numpy as np

def ReadData( fileName ):
    """This function takes a filename as input, and returns a dataframe with
    raw data read from that file in a Pandas DataFrame.  The DataFrame index
    should be the year, month and day of the observation.  DataFrame headers
    should be "agency_cd", "site_no", "Date", "Discharge", "Quality". The 
    "Date" column should be used as the DataFrame index. The pandas read_csv
    function will automatically replace missing values with np.NaN, but needs
    help identifying other flags used by the USGS to indicate no data is 
    availabiel.  Function returns the completed DataFrame, and a dictionary 
    designed to contain all missing value counts that is initialized with
    days missing between the first and last date of the file."""
    
    # define column names
    colNames = ['agency_cd', 'site_no', 'Date', 'Discharge', 'Quality']

    # open and read the file
    DataDF = pd.read_csv(fileName, header=1, names=colNames,  
                         delimiter=r"\s+",parse_dates=[2], comment='#',
                         na_values=['Eqp'])
    DataDF = DataDF.set_index('Date')
    
    # quantify the number of missing values
    MissingValues = DataDF["Discharge"].isna().sum()
   
    # Gross error check for negative discharge values
    DataDF['Discharge'][(DataDF['Discharge']<0)]=np.nan
    
    return( DataDF, MissingValues )

def ClipData( DataDF, startDate, endDate ):
    """This function clips the given time series dataframe to a given range 
    of dates. Function returns the clipped dataframe and and the number of 
    missing values."""
    
    # start date = October 1, 1969; enddate = September 30, 2019.
    # 50 water years of streamflow data for the analysis
    # clip the given streamflow timeseries for a given range of startdate to enddate
    DataDF = DataDF.loc[startDate:endDate]
    
    # quantifying the missing values 
    MissingValues = DataDF["Discharge"].isna().sum()
    
    return( DataDF, MissingValues )

def CalcTqmean(Qvalues):
    """This function computes the Tqmean of a series of data, typically
       a 1 year time series of streamflow, after filtering out NoData
       values.  Tqmean is the fraction of time that daily streamflow
       exceeds mean streamflow for each year. Tqmean is based on the
       duration rather than the volume of streamflow. The routine returns
       the Tqmean value for the given data array."""
       
    # Filtering No data value in the time series of streamflow   
    Qvalues = Qvalues.dropna()  
    
    # the length of the time series
    T_total = len(Qvalues)
    
    # fraction of time that daily streamflow exceeds mean streamflow for each year
    Tqmean = ((Qvalues > Qvalues.mean()).sum()/T_total)
    
    return ( Tqmean )

def CalcRBindex(Qvalues):
    """This function computes the Richards-Baker Flashiness Index
       (R-B Index) of an array of values, typically a 1 year time
       series of streamflow, after filtering out the NoData values.
       The index is calculated by dividing the sum of the absolute
       values of day-to-day changes in daily discharge volumes
       (pathlength) by total discharge volumes for each year. The
       routine returns the RBindex value for the given data array."""
   
    # Filtering No data value in the time series of streamflow
    Qvalues = Qvalues.dropna() 
    
    # Change in the daily discharge volumes
    diff_Q = Qvalues.diff()
   
    # sum of the absolute values of day-to-day changes 
    # in daily discharge volumes (pathlength)
    sum_diff_Q=abs(diff_Q).sum()
   
    # total discharge volumes for each year
    total_Q = Qvalues.sum()
   
    #RBindex value for the given data array
    RBindex = sum_diff_Q/total_Q
    
    return ( RBindex )

def Calc7Q(Qvalues):
    """This function computes the seven day low flow of an array of 
       values, typically a 1 year time series of streamflow, after 
       filtering out the NoData values. The index is calculated by 
       computing a 7-day moving average for the annual dataset, and 
       picking the lowest average flow in any 7-day period during
       that year.  The routine returns the 7Q (7-day low flow) value
       for the given data array."""
       
    # Filtering No data value in the time series of streamflow
    Qvalues=Qvalues.dropna()
   
    # 7-day moving average for the annual dataset
    # picking the lowest average flow in any 7-day period during that year. 
    val7Q=Qvalues.rolling(window=7).mean().min()
    
    return ( val7Q )

def CalcExceed3TimesMedian(Qvalues):
    """This function computes the number of days with flows greater 
       than 3 times the annual median flow. The index is calculated by 
       computing the median flow from the given dataset (or using the value
       provided) and then counting the number of days with flow greater than 
       3 times that value.   The routine returns the count of events greater 
       than 3 times the median annual flow value for the given data array."""
    
    # Filtering No data value in the time series of streamflow
    Qvalues=Qvalues.dropna()
      
    # count of events > 3 times the median annual flow value for the given data array
    median3x = (Qvalues > 3*Qvalues.median()).sum()
    
    return ( median3x )

def GetAnnualStatistics(DataDF):
    """This function calculates annual descriptive statistcs and metrics for 
    the given streamflow time series.  Values are retuned as a dataframe of
    annual values for each water year.  Water year, as defined by the USGS,
    starts on October 1."""
    
    # define column names for the dataframe
    colNames = ['site_no','Mean Flow','Peak Flow','Median','Coeff Var','Skew','TQmean','R-B Index','7Q','3xMedian']
    
    # Water Year is from Oct 01 to Sep 30 (USGS definition)
    Water_Year = DataDF.resample('AS-OCT')
    
    # Creating dataframe of annual statistic values for each water year
    data_annual = Water_Year.mean()
    WYDataDF=pd.DataFrame(0,index=data_annual.index,columns=colNames)
    
    # Providing statistics to fill the WYDataDF dataframe.
    # mean value of the site number
    WYDataDF['site_no']=Water_Year['site_no'].mean()
    
    # mean value of streamflow discharge for the water year
    WYDataDF['Mean Flow']=Water_Year['Discharge'].mean()
    
    # maximum value of streamflow discharge for the water year
    WYDataDF['Peak Flow']=Water_Year['Discharge'].max()
    
    # median value of streamflow discharge for the water year
    WYDataDF['Median']=Water_Year['Discharge'].median()
    
    # coefficient of variation(st. deviation/mean) of streamflow discharge for the water year
    WYDataDF['Coeff Var']=(Water_Year['Discharge'].std()/Water_Year['Discharge'].mean())*100
    
    # skewness in the streamflow discharge for the water year
    WYDataDF['Skew']=Water_Year.apply({'Discharge':lambda x: stats.skew(x)},raw=True)
    
    # Tqmean(fraction of time that daily streamflow exceeds mean streamflow for each year) 
    # of streamflow for the water year computed with CalcTqmean function above.
    WYDataDF['TQmean']=Water_Year.apply({'Discharge': lambda x: CalcTqmean(x)})
    
    # R-B Index(sum of the absolute values of day-to-day changes in daily discharge 
    # volumes/total discharge volumes for each year) of streamflow for the water year 
    # computed with CalcRBindex function above.
    WYDataDF['R-B Index']=Water_Year.apply({'Discharge': lambda x: CalcRBindex(x)})
    
    # seven day low flow of streamflow for the water year 
    # computed with Calc7Q function above.
    WYDataDF['7Q']=Water_Year.apply({'Discharge': lambda x: Calc7Q(x)})
    
    # number of days with flows > 3 times the annual median flow of streamflow for the water year 
    # computed with CalcExceed3TimesMedian function above.
    WYDataDF['3xMedian']=Water_Year.apply({'Discharge': lambda x: CalcExceed3TimesMedian(x)})
    
    return ( WYDataDF )

def GetMonthlyStatistics(DataDF):
    """This function calculates monthly descriptive statistics and metrics 
    for the given streamflow time series.  Values are returned as a dataframe
    of monthly values for each year."""
    
    # define column names for the dataframe
    colNames = ['site_no','Mean Flow','Coeff Var','TQmean','R-B Index']
    
    # Monthly distribution of the streamflow timeseries.
    Month_dist = DataDF.resample('M')
    
    # Creating dataframe of monthly descriptive statistics and metrics for the given streamflow time series
    data_monthly = Month_dist.mean()
    MoDataDF=pd.DataFrame(0,index=data_monthly.index,columns=colNames)
    
    # Providing statistics to fill the MoDataDF dataframe.
    # mean value of the site number
    MoDataDF['site_no']=Month_dist['site_no'].mean()
    
    # mean value of streamflow discharge 
    MoDataDF['Mean Flow']=Month_dist['Discharge'].mean()
    
    # coefficient of variation(st. deviation/mean) of streamflow discharge
    MoDataDF['Coeff Var']=(Month_dist['Discharge'].std()/Month_dist['Discharge'].mean())*100
    
    # Tqmean(fraction of time that daily streamflow exceeds mean streamflow for each year) 
    # of streamflow, computed with CalcTqmean function above.
    MoDataDF['TQmean']=Month_dist.apply({'Discharge': lambda x: CalcTqmean(x)})
    
    # R-B Index(sum of the absolute values of day-to-day changes in daily discharge 
    # volumes/total discharge volumes for each year) of streamflow  
    # computed with CalcRBindex function above.
    MoDataDF['R-B Index']=Month_dist.apply({'Discharge': lambda x: CalcRBindex(x)})
        
    return ( MoDataDF )

def GetAnnualAverages(WYDataDF):
    """This function calculates annual average values for all statistics and
    metrics.  The routine returns an array of mean values for each metric
    in the original dataframe."""
    
    # Providing mean of the WYDataDF dataframe gives the annual averages of 
    # all the statistics for streamflow timeseries.
    AnnualAverages = WYDataDF.mean(axis=0)
    
    return( AnnualAverages )

def GetMonthlyAverages(MoDataDF):
    """This function calculates annual average monthly values for all 
    statistics and metrics.  The routine returns an array of mean values 
    for each metric in the original dataframe."""
    
    
    # define column names for the dataframe
    colNames = ['site_no','Mean Flow','Coeff Var','TQmean','R-B Index']
    
    # Creating dataframe of annual average monthly values statistics and metrics 
    MonthlyAverages = pd.DataFrame(0,index = range(1,13),columns = colNames)
    
    j = [3,4,5,6,7,8,9,10,11,0,1,2]
    index=0
        
    # Providing statistics to fill the MonthlyAverages dataframe
    for i in range(12):
        # mean value of the site number of MoDataDF dataframe
        MonthlyAverages.iloc[index,0]=MoDataDF['site_no'][::12].mean()
        
        # mean of mean value of streamflow discharge from MoDataDF dataframe
        MonthlyAverages.iloc[index,1]=MoDataDF['Mean Flow'][j[index]::12].mean()
       
        # mean value of the coefficient of variation (st. deviation/mean) of MoDataDF dataframe
        MonthlyAverages.iloc[index,2]=MoDataDF['Coeff Var'][j[index]::12].mean()
        
        # mean of Tqmean of streamflow from ModataDF dataframe
        MonthlyAverages.iloc[index,3]=MoDataDF['TQmean'][j[index]::12].mean()
       
        # mean of RBIndex from the MoDataDF dataframe
        MonthlyAverages.iloc[index,4]=MoDataDF['R-B Index'][j[index]::12].mean()
        index+=1
   
    return( MonthlyAverages )

# the following condition checks whether we are running as a script, in which 
# case run the test code, otherwise functions are being imported so do not.
# put the main routines from your code after this conditional check.

if __name__ == '__main__':

    # define filenames as a dictionary
    # NOTE - you could include more than jsut the filename in a dictionary, 
    #  such as full name of the river or gaging site, units, etc. that would
    #  be used later in the program, like when plotting the data.
    fileName = { "Wildcat": "WildcatCreek_Discharge_03335000_19540601-20200315.txt",
                 "Tippe": "TippecanoeRiver_Discharge_03331500_19431001-20200315.txt" }
    
    # define blank dictionaries (these will use the same keys as fileName)
    DataDF = {}
    MissingValues = {}
    WYDataDF = {}
    MoDataDF = {}
    AnnualAverages = {}
    MonthlyAverages = {}
    
    # process input datasets
    for file in fileName.keys():
        
        print( "\n", "="*50, "\n  Working on {} \n".format(file), "="*50, "\n" )
        
        DataDF[file], MissingValues[file] = ReadData(fileName[file])
        print( "-"*50, "\n\nRaw data for {}...\n\n".format(file), DataDF[file].describe(), "\n\nMissing values: {}\n\n".format(MissingValues[file]))
        
        # clip to consistent period
        DataDF[file], MissingValues[file] = ClipData( DataDF[file], '1969-10-01', '2019-09-30' )
        print( "-"*50, "\n\nSelected period data for {}...\n\n".format(file), DataDF[file].describe(), "\n\nMissing values: {}\n\n".format(MissingValues[file]))
        
        # calculate descriptive statistics for each water year
        WYDataDF[file] = GetAnnualStatistics(DataDF[file])
        
        # calcualte the annual average for each stistic or metric
        AnnualAverages[file] = GetAnnualAverages(WYDataDF[file])
        
        print("-"*50, "\n\nSummary of water year metrics...\n\n", WYDataDF[file].describe(), "\n\nAnnual water year averages...\n\n", AnnualAverages[file])

        # calculate descriptive statistics for each month
        MoDataDF[file] = GetMonthlyStatistics(DataDF[file])

        # calculate the annual averages for each statistics on a monthly basis
        MonthlyAverages[file] = GetMonthlyAverages(MoDataDF[file])
        
        print("-"*50, "\n\nSummary of monthly metrics...\n\n", MoDataDF[file].describe(), "\n\nAnnual Monthly Averages...\n\n", MonthlyAverages[file])
    
    # Creating 'csv' file for annual metrics    
    Wildcat_WY = WYDataDF['Wildcat']
    Wildcat_WY['Station'] = 'Wildcat'
    Tippecanoe_WY = WYDataDF['Tippe']
    Tippecanoe_WY['Station'] = 'Tippe'
    Wildcat_WY = Wildcat_WY.append(Tippecanoe_WY)
    
    Wildcat_WY.to_csv('Annual_Metrics.csv', sep =',', index =True)
    
    # Creating 'csv' file for monthly metrics 
    Wildcat_MO = MoDataDF['Wildcat']
    Wildcat_MO['Station'] = 'Wildcat'
    Tippecanoe_MO = MoDataDF['Tippe']
    Tippecanoe_MO['Station'] = 'Tippe'
    Wildcat_MO = Wildcat_MO.append(Tippecanoe_MO)
    
    Wildcat_MO.to_csv('Monthly_Metrics.csv', sep =',', index =True)
    
    # Creating TAB delimited 'txt' file for annual averages metrics
    Wildcat_avgA = AnnualAverages['Wildcat']
    Wildcat_avgA['Station'] = 'Wildcat'
    Tippecanoe_avgA = AnnualAverages['Tippe']
    Tippecanoe_avgA['Station'] = 'Tippe'
    Wildcat_avgA = Wildcat_avgA.append(Tippecanoe_avgA)
    
    Wildcat_avgA.to_csv('Average_Annual_Metrics.txt', sep='\t', index = True)
    
    # Creating TAB delimited 'txt' file for monthly averages metrics
    Wildcat_avgM = MonthlyAverages['Wildcat']
    Wildcat_avgM['Station'] = 'Wildcat'
    Tippecanoe_avgM = MonthlyAverages['Tippe']
    Tippecanoe_avgM['Station'] = 'Tippe'
    Wildcat_avgM = Wildcat_avgM.append(Tippecanoe_avgM)
    
    Wildcat_avgM.to_csv('Average_Monthly_Metrics.txt', sep='\t', index = True)