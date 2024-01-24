import torch

def whitening_coloring_transform(content_feature, style_feature, strength=1., eps=1e-5):
	f_c = content_feature
	f_s = style_feature

	assert len(f_c.shape) == len(f_s.shape) and len(f_s.shape) == 3, 'Does not support batch processing.'
	assert 0 <= strength <= 1, 'Stylization strength should be in the range [0, 1].'

	C_c, H_c, W_c = f_c.shape
	C_s, H_s, W_s = f_s.shape
	assert C_c == C_s, 'Number of channels should be the same for differently-sized features.'

	# flattening the features
	f_c_flat = f_c.view(C_c, -1)
	f_s_flat = f_s.view(C_s, -1)

	# centering the features around 0 mean
	M_c = torch.mean(f_c_flat, -1).unsqueeze(-1)
	M_s = torch.mean(f_s_flat, -1).unsqueeze(-1)
	f_c_flat = f_c_flat - M_c
	f_s_flat = f_s_flat - M_s

	# SVD for covaraince matrices of features
	cov_c = (f_c_flat @ f_c_flat.T) / (H_c * W_c - 1)
	cov_s = (f_s_flat @ f_s_flat.T) / (H_s * W_s - 1)
	_, S_c, V_c = torch.svd(cov_c, some=False)
	_, S_s, V_s = torch.svd(cov_s, some=False)

	# Eigenvalues of covariance matrices of features
	n_evalues_c = C_c
	n_evalues_s = C_s

	for i in range(C_c):
		if S_c[i] < eps:
			n_evalues_c = i
			break
	for i in range(C_s):
		if S_s[i] < eps:
			n_evalues_s = i
	
	D_c = torch.diag((S_c[:n_evalues_c]).pow(-0.5))
	D_s = torch.diag((S_s[:n_evalues_s]).pow(0.5))

	# whitening transform
	f_c_whitened = V_c[:, :n_evalues_c] @ D_c
	f_c_whitened = f_c_whitened @ V_c[:, :n_evalues_c].T
	f_c_whitened = f_c_whitened @ f_c_flat

	# coloring transform
	f_c_colored = V_s[:, :n_evalues_s] @ D_s
	f_c_colored = f_c_colored @ V_s[:, :n_evalues_s].T
	f_c_colored = f_c_colored @ f_c_whitened

	# recentering around the mean of the style image
	f_cs_flat = f_c_colored + M_s

	# reshaping the stylized feature
	f_cs = f_cs_flat.view(C_c, H_c, W_c)

	# applying stylization strength
	stylized = strength * f_cs + (1.0 - strength) * f_c

	return stylized
