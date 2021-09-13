
import pandas as pd
from tqdm.auto import tqdm

print('Reading Data...')
# all necessary data has to be in a dedicated 'data' folder
data = pd.read_csv('../data/partially_combined/data_geo_hosp.csv')

# save data for later sanity check
data_length = len(data)
education = pd.read_csv('../data/county_data/Education.csv', encoding = "latin")
population = pd.read_csv('../data/county_data/PopulationEstimates.csv', encoding = "latin")
unemployment = pd.read_csv('../data/county_data/Unemployment.csv', encoding = "latin")
poverty = pd.read_csv('../data/county_data/PovertyEstimates.csv', encoding = "latin")
voters = pd.read_csv('../data/county_data/countypres_2000-2020.csv', encoding = "latin")

print('Reducing Voters Data...')
voters = voters[voters['year'] == 2020]
voters = voters[voters['county_fips'].notna()]
voters = voters.reset_index(drop=True)

print('Sorting Data by FIPS Code...')
# sort data
data = data.sort_values("county_fips_code")
education = education.sort_values("FIPS Code")
population = population.sort_values("FIPStxt")
unemployment = unemployment.sort_values("FIPS_Code")
poverty = poverty.sort_values("FIPStxt")
voters = voters.sort_values("county_fips")

def fips_lookup(fips_column, current_index, fips, data):
    for idx, fipsi in enumerate(fips_column[current_index:]):
        if fipsi == fips:
            return data[idx+current_index]
    return "NaN"

def fips_lookup2(column, fips, type_column, type_name, data):
    for idx, fipsi in enumerate(column):
        if (fipsi == fips):
            for i in range(len(type_column[idx:])):
                if type_column[idx+i] == type_name:
                    return data[idx+i]
    return "NaN"

def fips_lookup3(df, column, fips, type_column, type_column2, type_name, type_name2, data1, data2):
    for idx, fipsi in enumerate(column):
        if (fipsi == fips):
            i = 0
            while (df.iloc[idx+i]['county_fips'] == fips):
                if ((df.iloc[idx+i][type_column] == type_name) & (df.iloc[idx+i][type_column2] == type_name2)):
                    if((df.iloc[idx+i][data1] == "NaN") or (df.iloc[idx+i][data2] == "NaN")):
                        return "NaN"
                    ratio = df.iloc[idx+i][data1]/df.iloc[idx+i][data2]
                    return ratio
                i += 1
            i = 0
            count = 0
            while df.iloc[idx+i]['county_fips'] == fips:
                if df.iloc[idx+i][type_column] == type_name:
                    count += df.iloc[idx+i][data1]
                i += 1
            i = 0
            ratio = count/df.iloc[idx+i][data2]
            if count == 0:
                break
            else:
                return ratio
    return float(0)

print('Combining...')
# perform combination of data sets (takes ~1 minute on Intel® Core™ i5-5300U CPU @ 2.30GHz × 4 )
current_fips = data["county_fips_code"].iloc[0]
old_fips = 0
current_index = 0
pop1_list = [None] * len(data)
pop2_list = [None] * len(data)
pop3_list = [None] * len(data)
pop4_list = [None] * len(data)
edu1_list = [None] * len(data)
edu2_list = [None] * len(data)
edu3_list = [None] * len(data)
edu4_list = [None] * len(data)
empl1_list = [None] * len(data)
empl2_list = [None] * len(data)
empl3_list = [None] * len(data)
pov1_list = [None] * len(data)
vot1_list = [None] * len(data)
vot2_list = [None] * len(data)
vot3_list = [None] * len(data)
vot4_list = [None] * len(data)
vot5_list = [None] * len(data)

for idx, fips in enumerate(tqdm(data["county_fips_code"])):
    current_fips = fips
    
    # whenever fips changes, we cache the necessary census data
    if (current_fips != old_fips):
        
        pop1 = fips_lookup(population["FIPStxt"], current_index, fips, population["Rural-urban_Continuum Code_2013"])
        pop2 = fips_lookup(population["FIPStxt"], current_index, fips, population["Urban_Influence_Code_2013"])
        pop3 = fips_lookup(population["FIPStxt"], current_index, fips, population["Economic_typology_2015"])
        pop4 = fips_lookup(population["FIPStxt"], current_index, fips, population["POP_ESTIMATE_2019"])
        
        edu1 = fips_lookup(education["FIPS Code"], current_index, fips, education["Percent of adults with less than a high school diploma, 2015-19"])
        edu2 = fips_lookup(education["FIPS Code"], current_index, fips, education["Percent of adults with a high school diploma only, 2015-19"])
        edu3 = fips_lookup(education["FIPS Code"], current_index, fips, education["Percent of adults completing some college or associate's degree, 2015-19"])
        edu4 = fips_lookup(education["FIPS Code"], current_index, fips, education["Percent of adults with a bachelor's degree or higher, 2015-19"])
        
        empl1 = fips_lookup2(unemployment["FIPS_Code"], fips, unemployment["Attribute"], "Unemployment_rate_2020", unemployment["Value"])
        empl2 = fips_lookup2(unemployment["FIPS_Code"], fips, unemployment["Attribute"], "Median_Household_Income_2019", unemployment["Value"])
        empl3 = fips_lookup2(unemployment["FIPS_Code"], fips, unemployment["Attribute"], "Med_HH_Income_Percent_of_State_Total_2019", unemployment["Value"])
       
        pov1 = fips_lookup2(poverty["FIPStxt"], fips, poverty["Attribute"], "PCTPOVALL_2019", poverty["Value"])
        
        vot1 = fips_lookup3(voters, voters["county_fips"], fips, "party", "mode", "DEMOCRAT", "TOTAL", "candidatevotes", "totalvotes")
        vot2 = fips_lookup3(voters, voters["county_fips"], fips, "party", "mode", "REPUBLICAN", "TOTAL", "candidatevotes", "totalvotes")
        vot3 = fips_lookup3(voters, voters["county_fips"], fips, "party", "mode", "GREEN", "TOTAL", "candidatevotes", "totalvotes")
        vot4 = fips_lookup3(voters, voters["county_fips"], fips, "party", "mode", "LIBERTARIAN", "TOTAL", "candidatevotes", "totalvotes")
        vot5 = fips_lookup3(voters, voters["county_fips"], fips, "party", "mode", "OTHER", "TOTAL", "candidatevotes", "totalvotes")
        
        current_index += 1
    
    pop1_list[idx] = pop1
    pop2_list[idx] = pop2
    pop3_list[idx] = pop3
    pop4_list[idx] = pop4
    edu1_list[idx] = edu1
    edu2_list[idx] = edu2
    edu3_list[idx] = edu3
    edu4_list[idx] = edu4
    empl1_list[idx] = empl1
    empl2_list[idx] = empl2
    empl3_list[idx] = empl3
    pov1_list[idx] = pov1
    vot1_list[idx] = vot1
    vot2_list[idx] = vot2
    vot3_list[idx] = vot3
    vot4_list[idx] = vot4
    vot5_list[idx] = vot5

    old_fips = current_fips


print('Finalizing...')
data["rural-urban_Continuum Code_2013"] = pop1_list
data["urban_influence_code_2013"] = pop2_list
data["economic_typology_2015"] = pop3_list
data["pop_estimate_2019"] = pop4_list
data["less_than_hs_ratio"] = edu1_list
data["hs_only_ratio"] = edu2_list
data["some_college_ratio"] = edu3_list
data["bachelor_or_higher_ratio"] = edu4_list
data["unempl_rate"] = empl1_list
data["med_hh_income_19"] = empl2_list
data["med_hh_income_pct_state_total_19"] = empl3_list
data["poverty_19_ratio"] = pov1_list
data["democrat_20_ratio"] = vot1_list
data["republican_20_ratio"] = vot2_list
data["green_20_ratio"] = vot3_list
data["libertarian_20_ratio"] = vot4_list
data["other_20_ratio"] = vot5_list

print('Cleaning Data Frame...')
data['sum'] = data.apply(lambda row: row.democrat_20_ratio 
                                   + row.republican_20_ratio
                                   + row.green_20_ratio
                                   + row.libertarian_20_ratio
                                   + row.other_20_ratio, axis=1)

data = data[data['sum'] != 0]
data = data[data['sum'].isna() == False]

data = data.drop('sum', 1)
data = data.reset_index(drop=True)

deleted_count = data_length - len(data)

print(f'Dataset reduced by {deleted_count} entries.')
print('Writing Dataset...')
data.to_csv('../data/data.csv', index=False)