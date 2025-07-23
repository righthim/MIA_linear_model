import numpy as np

# generate d-dimensional c-category data
# X|Y=c~mu_c*e_c+ noise
# mu_c~1+noise

def data_generation(dimension,cateogry,n_data_per_class,mu_per_class,noise_level=0.5):
    d=dimension
    c=cateogry
    n_data_total=c*n_data_per_class

    x=noise_level*np.random.randn(n_data_total,d)
    y=np.zeros(n_data_total)
    for category in range(c):
        x[category*n_data_per_class:category*n_data_per_class+n_data_per_class,category]+=mu_per_class[category]
        y[category*n_data_per_class:category*n_data_per_class+n_data_per_class]=category
    y=y 
    return x,y

# Overfitted model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

setting_num = 0
d=100
c=10
n_data_per_class=10
mu_per_class=0.5*np.random.randn(c)+np.ones(c)
np.save(f'mu_per_class_setting{setting_num}',mu_per_class)
np.save(f'd_setting{setting_num}',d)
np.save(f'c_setting{setting_num}',c)
np.save(f'n_data_per_class_setting{setting_num}',n_data_per_class)

n_data_total=c*n_data_per_class
num_exp=100
training_accuracy=np.zeros(num_exp)
test_accuracy=np.zeros(num_exp)
for i in range(num_exp):
    x_train_i,y_train_i=data_generation(d,c,n_data_per_class,mu_per_class)
    x_test_i,y_test_i=data_generation(d,c,n_data_per_class,mu_per_class)
    lda = LinearDiscriminantAnalysis(n_components=c-1)
    lda.fit(x_train_i,y_train_i)

    # training error
    training_pred=lda.predict(x_train_i)
    training_accuracy[i]=np.mean(training_pred==y_train_i)

    # test error
    test_pred=lda.predict(x_test_i)
    test_accuracy[i]=np.mean(test_pred==y_test_i)
print(f'training error: ({np.mean(training_accuracy)-1.96*np.std(training_accuracy)/np.sqrt(num_exp)}, {np.mean(training_accuracy)+1.96*np.std(training_accuracy)/np.sqrt(num_exp)})')
print(f'test error: ({np.mean(test_accuracy)-1.96*np.std(test_accuracy)/np.sqrt(num_exp)}, {np.mean(test_accuracy)+1.96*np.std(test_accuracy)/np.sqrt(num_exp)})')
print(f'training error vs test error gap: {np.mean(training_accuracy)-np.mean(test_accuracy)}')

# Attack
## Target model
n_data_total=c*n_data_per_class
x_train,y_train=data_generation(d,c,n_data_per_class,mu_per_class)
x_test,y_test=data_generation(d,c,n_data_per_class,mu_per_class)
np.save(f'target_training_data_setting{setting_num}',x_train)
np.save(f'target_training_label_setting{setting_num}',y_train)
np.save(f'target_test_data_setting{setting_num}',x_test)
np.save(f'target_test_label_setting{setting_num}',y_test)

target_model = LinearDiscriminantAnalysis(n_components=c-1)
target_model.fit(x_train,y_train)

## Shadow model
import joblib
num_shadow=100
for i in range(num_shadow):
    x_train_i,y_train_i=data_generation(d,c,n_data_per_class,mu_per_class)
    x_test_i,y_test_i=data_generation(d,c,n_data_per_class,mu_per_class)
    lda_i = LinearDiscriminantAnalysis(n_components=c-1)
    lda_i.fit(x_train_i,y_train_i)
    
    # Save the model
    joblib.dump(lda_i, f'shadow_model{i}_setting{setting_num}.pkl')
    np.save(f'shadow_training_data{i}_setting{setting_num}',x_train_i)
    np.save(f'shadow_training_labels{i}_setting{setting_num}',y_train_i)
    np.save(f'shadow_test_data{i}_setting{setting_num}',x_test_i)
    np.save(f'shadow_test_labels{i}_setting{setting_num}',y_test_i)

## Generate training dataset for the attack model
attack_training_data_list=[]
attack_training_labels_list=[]
membership_labels_list=[]
for i in range(num_shadow):
    shadow_model_i=joblib.load(f'shadow_model{i}_setting{setting_num}.pkl')
    shadow_training_data_i=np.load(f'shadow_training_data{i}_setting{setting_num}.npy')
    training_pred_i=shadow_model_i.predict_log_proba(shadow_training_data_i)
    attack_in_data_i=np.concatenate((shadow_training_data_i,training_pred_i),axis=1)
    attack_training_data_list.append(attack_in_data_i)
    attack_training_labels_list.append(np.load(f'shadow_training_labels{i}_setting{setting_num}.npy'))
    membership_labels_list.append(np.ones(n_data_total))

    shadow_test_data_i=np.load(f'shadow_test_data{i}_setting{setting_num}.npy')
    test_pred_i=shadow_model_i.predict_log_proba(shadow_test_data_i)
    attack_out_data_i=np.concatenate((shadow_test_data_i,test_pred_i),axis=1)
    attack_training_data_list.append(attack_out_data_i)
    attack_training_labels_list.append(np.load(f'shadow_test_labels{i}_setting{setting_num}.npy'))
    membership_labels_list.append( np.zeros(n_data_total))
attack_training_data=np.concatenate(attack_training_data_list,axis=0)
attack_training_labels=np.concatenate(attack_training_labels_list)
attack_membership_labels=np.concatenate(membership_labels_list) 
for i in range(c):
    np.save(f'attack_training_data_class{i}_setting{setting_num}',attack_training_data[attack_training_labels==i,:])
    np.save(f'attack_training_membership_class{i}_setting{setting_num}',attack_membership_labels[attack_training_labels==i])

## Train attack models for each class i
for i in range(c):
    print(f'Attack on class {i} start')
    training_data=np.load(f'attack_training_data_class{i}_setting{setting_num}.npy')
    training_membership=np.load(f'attack_training_membership_class{i}_setting{setting_num}.npy')
    
    lda = LinearDiscriminantAnalysis()
    lda.fit(training_data,training_membership)
    joblib.dump(lda,f'attack_model_lda_class{i}_setting{setting_num}.pkl')

    logi=LogisticRegression()
    logi.fit(training_data,training_membership)
    joblib.dump(logi,f'attack_model_logi_class{i}_setting{setting_num}.pkl')
    print(f'Attack on class {i} end')

## Evaluate the performance of the attack models
lda_pred_membership_list=[]
logi_pred_membership_list=[]
membership_list=[]
for i in range(c):
    attack_model_lda_i=joblib.load(f'attack_model_lda_class{i}_setting{setting_num}.pkl')
    attack_model_logi_i=joblib.load(f'attack_model_logi_class{i}_setting{setting_num}.pkl')

    x_training_i=x_train[y_train==i,:]
    pred_training_i=target_model.predict_proba(x_training_i)
    data_training_i=np.concatenate((x_training_i,pred_training_i),axis=1)

    x_test_i=x_test[y_test==i,:]
    pred_test_i=target_model.predict_proba(x_test_i)
    data_test_i=np.concatenate((x_test_i,pred_test_i),axis=1)
    candidate_data_i=np.concatenate((data_training_i,data_test_i),axis=0)
    membership_i=np.concatenate((np.ones(x_training_i.shape[0]), np.zeros(x_test_i.shape[0])),axis=0) 
    membership_list.append(membership_i)

    lda_pred_membership_i=attack_model_lda_i.predict_proba(candidate_data_i)
    logi_pred_membership_i=attack_model_logi_i.predict_proba(candidate_data_i)
    lda_pred_membership_list.append(lda_pred_membership_i)
    logi_pred_membership_list.append(logi_pred_membership_i)

lda_pred_membership=np.concatenate(lda_pred_membership_list,axis=0)
logi_pred_membership=np.concatenate(logi_pred_membership_list,axis=0)
membership=np.concatenate(membership_list,axis=0)
np.save(f'lda_pred_membership_setting{setting_num}',lda_pred_membership)
np.save(f'logi_pred_membership_setting{setting_num}',logi_pred_membership)
np.save(f'membership_setting{setting_num}',membership)


from scipy.stats import norm

x=np.concatenate((x_train,x_test),axis=0)
y=np.concatenate((y_train,y_test)).astype(int)
membership0=np.concatenate((np.ones(len(y_train)),np.zeros(len(y_test)))).astype(int)
prob = target_model.predict_proba(x)
p_obs = prob[np.arange(len(x)), y]
conf_obs=np.log(p_obs/(1+1e-6-p_obs))
n=x.shape[0]
num_exp=1000
conf=np.zeros((num_exp,n))
mu_in=np.zeros(n)
mu_out=np.zeros(n)
sigma_in=np.zeros(n)
sigma_out=np.zeros(n)
membership_pred=np.zeros(n)

inclusion_matrix=np.zeros((num_exp, n), dtype=bool)
for i in range(n):
    idx=np.random.choice(num_exp,size=num_exp//2,replace=False)
    inclusion_matrix[:, i] = False
    inclusion_matrix[idx, i] = True

for j in range(num_exp):
    data_j,label_j=data_generation(d,c,n_data_per_class,mu_per_class)
    attack_data_j=np.concatenate((data_j,x[inclusion_matrix[j,:],:]),axis=0)
    attack_label_j=np.concatenate((label_j,y[inclusion_matrix[j,:]]))
    attack_model_j = LinearDiscriminantAnalysis(n_components=c-1)
    attack_model_j.fit(attack_data_j,attack_label_j)
    prob_j=attack_model_j.predict_proba(x)
    p_j=prob_j[np.arange(len(x)),y]
    conf_j=np.log(p_j/(1+1e-6-p_j))
    conf[j,:]=conf_j
for i in range(n):
    conf_in_i=conf[inclusion_matrix[:,i],i]
    conf_out_i=conf[~inclusion_matrix[:,i],i]
    mu_in[i]=np.median(conf_in_i)
    mu_out[i]=np.median(conf_out_i)
    sigma_in[i]=np.std(conf_in_i)
    sigma_out[i]=np.std(conf_out_i)
    Lambda_i=norm.logpdf(conf_obs[i],loc=mu_in[i],scale=sigma_in[i])-norm.logpdf(conf_obs[i],loc=mu_out[i],scale=sigma_out[i])
    membership_pred[i]=Lambda_i
np.save(f'membership_likelihood_setting{setting_num}',membership_pred)
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
lda_fpr, lda_tpr, lda_thresholds = roc_curve(membership, lda_pred_membership[:,1])
logi_fpr, logi_tpr, logi_thresholds = roc_curve(membership, logi_pred_membership[:,1])
fpr, tpr, thresholds = roc_curve(membership0, membership_pred,pos_label=1)
# Plot
plt.figure()
plt.plot(fpr, tpr,label='likelihood attack')
plt.plot(lda_fpr, lda_tpr,label='lda attack')
plt.plot(logi_fpr, logi_tpr,label='logistic attack')
plt.plot([0, 1], [0, 1], 'k--')  # diagonal
plt.xscale('log')
plt.yscale('log')
plt.xticks([10**(-i) for i in range(6)])
plt.yticks([10**(-i) for i in range(6)])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.savefig(f'attack_setting{setting_num}.jpg')
plt.show()

