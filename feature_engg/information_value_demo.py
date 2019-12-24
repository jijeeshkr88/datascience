import numpy as np
import pandas as pd









def calculate_woe_iv(dataset, feature, target):
    lst = []
    for i in range(dataset[feature].nunique()):
        val = list(dataset[feature].unique())[i]
        lst.append({
            'Value': val,
            'All': dataset[dataset[feature] == val].count()[feature],
            'Good': dataset[(dataset[feature] == val) & (dataset[target] == 0)].count()[feature],
            'Bad': dataset[(dataset[feature] == val) & (dataset[target] == 1)].count()[feature]
        })
        
    dset = pd.DataFrame(lst)
    print(dset)
    dset['Distr_Good'] = dset['Good'] / dset['Good'].sum()
    dset['Distr_Bad'] = dset['Bad'] / dset['Bad'].sum()
    dset['WoE'] = np.log(dset['Distr_Good'] / dset['Distr_Bad'])
    dset = dset.replace({'WoE': {np.inf: 0, -np.inf: 0}})
    dset['IV'] = (dset['Distr_Good'] - dset['Distr_Bad']) * dset['WoE']
    iv = dset['IV'].sum()
    dset = dset.sort_values(by='WoE')
    return dset, iv

def iv_calc(var_list,target):
	unique_vals = np.unique(var_list)
	iv_details = []
	var_iv = 0
	total_bad = sum(target)
	total_good = len(target)-total_bad
	for val in unique_vals:
		d = {}
		val_indices = np.where(var_list==val)[0]
		val_target = target[val_indices]
		val_count = len(val_target)
		val_bad_count = sum(val_target)
		val_good_count = val_count-val_bad_count
		val_good_distbn = val_good_count/total_good
		val_bad_distbn = val_bad_count/total_bad
		val_woe = np.log(val_good_distbn/val_bad_distbn)
		val_iv = (val_good_distbn-val_bad_distbn)*val_woe
		if val_woe in [-np.inf,np.inf]:
			val_woe = 0
		d["value"] = val
		d["all"] = val_count
		d["bad"] = val_bad_count
		d["good"] = val_good_count
		d["goodDistbn"] = val_good_distbn
		d["badDistbn"] = val_bad_distbn
		d["woe"] = val_woe
		d["iv"] = val_iv
		var_iv+=val_iv
		iv_details.append(d)
	return var_iv,iv_details


if __name__ == '__main__':
	import time
	file_name = "./data/Churn_Modelling.csv"
	df = pd.read_csv(file_name)
	target = np.array(df["Exited"].tolist())
	var_list = np.array(df["Gender"].tolist())
	tic1 = time.time()
	ivdf,iv = calculate_woe_iv(df,'Gender','Exited')
	print("pandas time:",time.time()-tic1)
	tic2 = time.time()
	niv,iv_det = iv_calc(var_list,target)
	print("numpy time:",time.time()-tic2)