from one.api import ONE
from datetime import datetime
import pandas as pd
import numpy as np
from disentangled_rnns.library import rnn_utils

one = ONE(
					mode="remote",
					base_url="https://openalyx.internationalbrainlab.org",
					password="international",
					silent=True,
        )
TAG = '2025_Q3_Zang_et_al_Aging'

def get_session_df(TAG):
	sessions = one.search(tag=TAG)
	full_frame=0
	for eid in [sessions[0]]:
		trials = one.load_object(eid, 'trials')
		trials.pop('intervals_bpod',None)
		trials.pop('intervals',None)
		session_details = one.get_details(eid)
		subject = session_details['subject']
		sess_date = session_details['start_time'][:10]
		trials = pd.DataFrame.from_dict(trials)
		trials['Subject'] = subject
		trials['Session_id'] = str(eid)
		try:
			subj = one.alyx.rest('subjects', 'list', nickname=subject)
			subj_dob = subj[0]['birth_date']
			session_sex = subj[0]['sex']
			session_weight = subj[0]['reference_weight']
			session_age = (datetime.strptime(sess_date, '%Y-%m-%d') \
										 - datetime.strptime(subj_dob, '%Y-%m-%d')).days
			trials['sex'] = session_sex
			trials['weight'] = session_weight
			trials['age'] = session_age
			trials['trialInSession'] = range(len(trials))
		except Exception:
				trials.age = np.nan
		if type(full_frame)==int:
				full_frame = trials
		else:
				full_frame = pd.concat([full_frame,trials], sort=False)
	full_frame['rt'] = full_frame['response_times'] \
	- full_frame['stimOnTrigger_times']
	return full_frame

def make_dataset_session_embedding(data, inputs, outputs, max_len=400):
	'''Makes a list of DatasetRNN to feed into a DisRNN.
	Inputs:
		- data (pandas.DataFrame): must have columns 'session', 'probabilityLeft',
		and anything in inputs and outputs
		- inputs (list of str): list of inputs to consider in the DisRNN
		- outputs (list of str): list of outputs to be predicted by the DisRNN
		- max_len (int): number of trials to consider in each session
	Returns:
		- datasets (list of DatasetRNN)
		- my_sessions (list of str): list of the sessions in order of appearence in
		datasets
		- blocks (ndarray): 
	datasets = []
	all_sessions = np.unique(data.session)
	my_sessions = []
	blocks=0
	for session in all_sessions:
		curdat = data[data.session == session]
		if max(curdat.trialInSession)>=max_len-1:
			my_sessions.append(session)
			xs = np.expand_dims(
					curdat[inputs].to_numpy()[0:max_len,:],
					axis=1)
			ys = np.expand_dims(curdat[outputs].to_numpy()[0:max_len,:],
													axis=1)

			if type(blocks)==int:
							blocks = np.expand_dims(curdat['probabilityLeft'].to_numpy()[0:max_len],
																			axis=1)
			else:
					cur_bl = np.expand_dims(
							curdat['probabilityLeft'].to_numpy()[0:max_len],
							axis=1)
					blocks = np.concatenate((blocks, cur_bl), axis=1)

			ys[:,:,0] = (ys[:,:,0]+1)
			# try removing (not fitting) missing choices
			ys[ys==1] = -2
			ys[ys==2] = 1
			ds = rnn_utils.DatasetRNN(xs, ys, y_type='categorical',
																x_names=inputs, y_names=outputs,
																n_classes=2)
			datasets.append(ds)
	return datasets, my_sessions, blocks
