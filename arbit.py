def create_merged_df(filenames):
    count = 0
    for i in range((len(filenames))):
        arr_data = xr.open_dataset("../"+filenames[0][i])

        # Selecting only dates after MEOP 2018 release. 
        selTime = arr_data.JULD > np.datetime64("2018-01-14")
        nlev = len(arr_data['PRES_ADJUSTED'][0])
        nprof = len(arr_data['PRES_ADJUSTED'][selTime])
        if nprof > 0:
            if(count == 0):
                # creating unique serial number for each profile in NPROF
                NPROF = np.arange(nprof)
                NPROF = np.array([[NPROF[j]] * nlev for j in range(len(NPROF)) ])
            else:
                last_end = NPROF.flatten()[-1] + 1
                NPROF = np.arange(last_end, last_end+nprof )
                NPROF = np.array([[NPROF[j]] * nlev for j in range(len(NPROF)) ])

            ind = np.arange(nprof*nlev)
            # arr_data["LATITUDE"] has length of nprof. Now expanding for each row of the CSV file, so each profile having nlev rows will have the latitude of that profile repeated on each row.
            lat = np.array([[arr_data["LATITUDE"][selTime][j].values]* nlev for j in range(nprof) ])
            lon = np.array([[arr_data["LONGITUDE"][selTime][j].values]* nlev for j in range(nprof) ])
            posqc = np.array([[arr_data["POSITION_QC"][selTime][j].values]* nlev for j in range(nprof) ])
            juld = np.array([[arr_data["JULD"][selTime][j].values]* nlev for j in range(nprof) ])

            if(count == 0):    
                df = {'PLATFORM_NUMBER': pd.Series([str(arr_data["PLATFORM_NUMBER"][selTime][0].values)]*len(ind), index=ind), 
                      'PROFILE_NUMBER' : pd.Series(NPROF.flatten(), index=ind ),
                      'TEMP_ADJUSTED': pd.Series(arr_data["TEMP_ADJUSTED"][selTime].values.flatten(), index=ind), 
                      'PSAL_ADJUSTED': pd.Series(arr_data["PSAL_ADJUSTED"][selTime].values.flatten(), index=ind), 
                      'PRES_ADJUSTED': pd.Series(arr_data["PRES_ADJUSTED"][selTime].values.flatten(), index=ind),  
                      'PRES_ADJUSTED_QC': pd.Series(arr_data["PRES_ADJUSTED_QC"][selTime].values.flatten(), index=ind), 
                      'PRES_ADJUSTED_ERROR':pd.Series(arr_data["PRES_ADJUSTED_ERROR"][selTime].values.flatten(), index=ind), 
                      'TEMP_ADJUSTED_QC': pd.Series(arr_data["TEMP_ADJUSTED_QC"][selTime].values.flatten(), index=ind), 
                      'TEMP_ADJUSTED_ERROR': pd.Series(arr_data["TEMP_ADJUSTED_ERROR"][selTime].values.flatten(), index=ind), 
                      'PSAL_ADJUSTED_QC': pd.Series(arr_data["PSAL_ADJUSTED_QC"][selTime].values.flatten(), index=ind),  
                      'PSAL_ADJUSTED_ERROR':pd.Series(arr_data["PSAL_ADJUSTED_ERROR"][selTime].values.flatten(), index=ind),  
                      'JULD': pd.Series(juld.flatten(), index=ind),  
                      'LATITUDE': pd.Series(lat.flatten(), index=ind),  
                      'LONGITUDE': pd.Series(lon.flatten(), index=ind),  
                      'POSITION_QC': pd.Series(posqc.flatten(), index=ind) }
                df = pd.DataFrame(df)
            else:
                df_i = {'PLATFORM_NUMBER': pd.Series([str(arr_data["PLATFORM_NUMBER"][selTime][0].values)]*len(ind), index=ind), 
                      'PROFILE_NUMBER' : pd.Series(NPROF.flatten(), index=ind ),
                      'TEMP_ADJUSTED': pd.Series(arr_data["TEMP_ADJUSTED"][selTime].values.flatten(), index=ind), 
                      'PSAL_ADJUSTED': pd.Series(arr_data["PSAL_ADJUSTED"][selTime].values.flatten(), index=ind), 
                      'PRES_ADJUSTED': pd.Series(arr_data["PRES_ADJUSTED"][selTime].values.flatten(), index=ind),  
                      'PRES_ADJUSTED_QC': pd.Series(arr_data["PRES_ADJUSTED_QC"][selTime].values.flatten(), index=ind), 
                      'PRES_ADJUSTED_ERROR':pd.Series(arr_data["PRES_ADJUSTED_ERROR"][selTime].values.flatten(), index=ind), 
                      'TEMP_ADJUSTED_QC': pd.Series(arr_data["TEMP_ADJUSTED_QC"][selTime].values.flatten(), index=ind), 
                      'TEMP_ADJUSTED_ERROR': pd.Series(arr_data["TEMP_ADJUSTED_ERROR"][selTime].values.flatten(), index=ind), 
                      'PSAL_ADJUSTED_QC': pd.Series(arr_data["PSAL_ADJUSTED_QC"][selTime].values.flatten(), index=ind),  
                      'PSAL_ADJUSTED_ERROR':pd.Series(arr_data["PSAL_ADJUSTED_ERROR"][selTime].values.flatten(), index=ind),  
                      'JULD': pd.Series(juld.flatten(), index=ind),  
                      'LATITUDE': pd.Series(lat.flatten(), index=ind),  
                      'LONGITUDE': pd.Series(lon.flatten(), index=ind),  
                      'POSITION_QC': pd.Series(posqc.flatten(), index=ind)  }
                df_i = pd.DataFrame(df_i)

                #pdb.set_trace()
                df = df.append(df_i, ignore_index=True)
                del(arr_data)
            count += 1
    return df
