import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Allocator(object):
    """
    The SparsityRatioAllocator for SNELL Model that will be called every training step. 

    Args:
        model: the model that we apply allocator to.
        init_ratio (`float`): The initial sparsity ratio for each incremental matrix.
        target_ratio (`float`): The target average sparsity ratio of incremental matrix.
        init_warmup (`int`): The steps of initial fine-tuning warmup.
        final_warmup (`int`): The step of final fine-tuning.
        mask_interval (`int`): The time internval between two budget allocations.
        beta1 (`float`): The hyperparameter of EMA for sensitivity smoothing.
        beta2 (`float`): The hyperparameter of EMA for undertainty quantification.
        total_step (`int`): The total training steps, correctly configured before training.
    """
    def __init__(
        self, model, 
        target_ratio:int, 
        init_warmup:int, 
        final_warmup:int,
        mask_interval:int,
        beta1:float, 
        beta2:float, 
        total_step=None, 
    ):
        
        self.initial_warmup = init_warmup
        self.final_warmup = final_warmup 
        self.mask_interval = mask_interval
        self.beta1 = beta1
        self.beta2 = beta2
        self.total_step = total_step

        self.model = model
        self.ipt = {} 
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}
        self.cat_ipt = {}
        self.ratio_pattern = {} 

        self.init_nonzero_param = 0
        self.target_nonzero_param = 0
        self.alloc_volume_dict, self.score_volume_dict = None, None

        self.name_list = [] 
        self.max_alloc_list = []
        for name, module in self.model.named_modules():
            if hasattr(module, "init_thres"): 
                self.name_list.append(name)
                self.max_alloc_list.append(module.in_features * module.out_features)
                self.init_nonzero_param += (1-module.init_thres) * module.in_features * module.out_features
                self.target_nonzero_param += (1-target_ratio) * module.in_features * module.out_features
        assert (self.beta1<1 and self.beta1>0)
        assert (self.beta2<1 and self.beta2>0)

    def set_total_step(self, total_step:int): 
        # Set total step number 
        self.total_step = total_step
        assert self.total_step>self.initial_warmup+self.final_warmup

    def get_sparsity_pattern(self):
        return self.alloc_volume_dict
    
    def get_score_pattern(self):
        return self.score_volume_dict

    def schedule_threshold(self, step:int):
        # Global budget schedule
        mask_ind = False 
        initial_warmup = self.initial_warmup 
        final_warmup = self.final_warmup 
        total_step = self.total_step 
        self.global_step = step
        if step <= initial_warmup: 
            # Initial warmup 
            curr_volume = self.init_nonzero_param 
            mask_ind = False 
        elif step > total_step - final_warmup: 
            # Final fine-tuning 
            curr_volume = self.target_nonzero_param 
            # Fix the ratio pattern by 
            # always masking the same unimportant singluar values 
            mask_ind = True 
        else: 
            # Budget decreasing 
            mul_coeff = 1-(step-initial_warmup)/(total_step-final_warmup-initial_warmup)
            curr_volume = self.target_nonzero_param + (self.init_nonzero_param-self.target_nonzero_param)*(mul_coeff**3)
            curr_volume = int(curr_volume)
            #print(mul_coeff, mul_coeff**3, curr_volume, self.target_nonzero_param)
            mask_ind = True if step % self.mask_interval == 0 else False 
        return curr_volume, mask_ind 


    def update_ipt(self, model, metric='ipt'): 
        for name, module in model.named_modules():
            if hasattr(module, "In_Qs"): 
                if f"{name}.In_Qs" not in self.ipt:
                    if metric == 'ipt':
                        self.ipt[f"{name}.In_Qs"] = torch.zeros_like(module.In_Qs)
                        self.ipt[f"{name}.Out_Qs"] = torch.zeros_like(module.Out_Qs)
                        self.exp_avg_ipt[f"{name}.In_Qs"] = torch.zeros_like(module.In_Qs)
                        self.exp_avg_ipt[f"{name}.Out_Qs"] = torch.zeros_like(module.Out_Qs)
                        self.exp_avg_unc[f"{name}.In_Qs"] = torch.zeros_like(module.In_Qs)
                        self.exp_avg_unc[f"{name}.Out_Qs"] = torch.zeros_like(module.Out_Qs)
                    elif metric == 'w_mag':
                        self.exp_avg_ipt[name] = 0
                        self.exp_avg_unc[name] = 0
                    elif metric == 'mag':
                        self.ipt[f"{name}.In_Qs"] = 0
                        self.ipt[f"{name}.Out_Qs"] = 0
                        self.exp_avg_ipt[f"{name}.In_Qs"] = 0
                        self.exp_avg_ipt[f"{name}.Out_Qs"] = 0
                        self.exp_avg_unc[f"{name}.In_Qs"] = 0
                        self.exp_avg_unc[f"{name}.Out_Qs"] = 0
                    # print(name)
                with torch.no_grad():
                    # Calculate sensitivity 
                    if metric == 'ipt':
                        # self.ipt[n] = (p * p.grad).abs().detach()
                        # print(torch.sum(torch.isnan(module.In_Qs.grad)))
                        # print(torch.sum(torch.isnan(module.Out_Qs.grad)))
                        self.ipt[f"{name}.In_Qs"] = (module.In_Qs * module.In_Qs.grad).abs().detach()
                        self.ipt[f"{name}.Out_Qs"] = (module.Out_Qs * module.Out_Qs.grad).abs().detach()
                        self.ipt[f"{name}.In_Qs"] = torch.where(torch.isnan(self.ipt[f"{name}.In_Qs"]), torch.full_like(self.ipt[f"{name}.In_Qs"], 0), self.ipt[f"{name}.In_Qs"])
                        self.ipt[f"{name}.Out_Qs"] = torch.where(torch.isnan(self.ipt[f"{name}.Out_Qs"]), torch.full_like(self.ipt[f"{name}.Out_Qs"], 0), self.ipt[f"{name}.Out_Qs"])
                        
                        self.exp_avg_ipt[f"{name}.In_Qs"] = self.beta1 * self.exp_avg_ipt[f"{name}.In_Qs"] + \
                                            (1-self.beta1)*self.ipt[f"{name}.In_Qs"]
                        self.exp_avg_ipt[f"{name}.Out_Qs"] = self.beta1 * self.exp_avg_ipt[f"{name}.Out_Qs"] + \
                                            (1-self.beta1)*self.ipt[f"{name}.Out_Qs"]
                        self.exp_avg_unc[f"{name}.In_Qs"] = self.beta2 * self.exp_avg_unc[f"{name}.In_Qs"] + \
                                            (1-self.beta2)*(self.ipt[f"{name}.In_Qs"]-self.exp_avg_ipt[f"{name}.In_Qs"]).abs()
                        self.exp_avg_unc[f"{name}.Out_Qs"] = self.beta2 * self.exp_avg_unc[f"{name}.Out_Qs"] + \
                                            (1-self.beta2)*(self.ipt[f"{name}.Out_Qs"]-self.exp_avg_ipt[f"{name}.Out_Qs"]).abs()        
                        # print('utilize ipt')
                        # print(self.ipt[f"{name}.In_Qs"].mean(), self.ipt[f"{name}.Out_Qs"].mean())
                    elif metric == 'mag':
                        # self.ipt[n] = p.abs().detach()
                        self.ipt[f"{name}.In_Qs"] = module.In_Qs.abs().detach()
                        self.ipt[f"{name}.Out_Qs"] = module.Out_Qs.abs().detach()
                        self.exp_avg_ipt[f"{name}.In_Qs"] = self.beta1 * self.exp_avg_ipt[f"{name}.In_Qs"] + \
                                            (1-self.beta1)*self.ipt[f"{name}.In_Qs"]
                        self.exp_avg_ipt[f"{name}.Out_Qs"] = self.beta1 * self.exp_avg_ipt[f"{name}.Out_Qs"] + \
                                            (1-self.beta1)*self.ipt[f"{name}.Out_Qs"]
                        self.exp_avg_unc[f"{name}.In_Qs"] = self.beta2 * self.exp_avg_unc[f"{name}.In_Qs"] + \
                                            (1-self.beta2)*(self.ipt[f"{name}.In_Qs"]-self.exp_avg_ipt[f"{name}.In_Qs"]).abs()
                        self.exp_avg_unc[f"{name}.Out_Qs"] = self.beta2 * self.exp_avg_unc[f"{name}.Out_Qs"] + \
                                            (1-self.beta2)*(self.ipt[f"{name}.Out_Qs"]-self.exp_avg_ipt[f"{name}.Out_Qs"]).abs()        
                    elif metric == 'w_mag':
                        self.ipt[name] = module.get_w_mag()
                        # self.ipt[f"{name}.In_Qs"] = module.get_w_mag()
                        # self.ipt[f"{name}.Out_Qs"] = module.get_w_mag()
                        # Update sensitivity 
                        self.exp_avg_ipt[name] = self.beta1 * self.exp_avg_ipt[name] + \
                                            (1-self.beta1)*self.ipt[name]
                        # Update uncertainty 
                        self.exp_avg_unc[name] = self.beta2 * self.exp_avg_unc[name] + \
                                            (1-self.beta2)*(self.ipt[name]-self.exp_avg_ipt[name]).abs()

    def calculate_score(self, n, p=None, module=None, metric="ipt"):
        if metric == "ipt":
            # Combine the senstivity and uncertainty 
            ipt_score = self.exp_avg_ipt[n] * self.exp_avg_unc[n]
        elif metric == "mag":
            # ipt_score = p.abs().detach().clone() 
            ipt_score = self.exp_avg_ipt[n] * self.exp_avg_unc[n]
        elif metric == "w_mag":
            # ipt_score = module.get_w_mag()
            ipt_score = self.exp_avg_ipt[n] * self.exp_avg_unc[n]
        else:
            raise ValueError("Unexpected Metric: %s"%metric)
        return ipt_score

    def param_alloc(self, scores, curr_volume, max_alloc_volumes):
        alloc_volumes = (curr_volume * scores).squeeze().int() # allocation according to scores
        # alloc_volumes = torch.min(max_alloc_volumes, alloc_volumes) # assigned volumes for each layer
        alloc_volumes = torch.min(max_alloc_volumes, alloc_volumes) # assigned volumes for each layer

        # print(alloc_volumes)
        alloc_delta = curr_volume - torch.sum(alloc_volumes) # remaining volumes
        while alloc_delta > 0:
            volume_remain = max_alloc_volumes - alloc_volumes # assignable volumes for each layer
            # print('volume_remain', volume_remain)
            assignable_idxs = torch.nonzero(volume_remain).squeeze()
            alloc_volumes[assignable_idxs] += torch.min((alloc_delta * scores[assignable_idxs] / torch.sum(scores[assignable_idxs])).squeeze(), volume_remain[assignable_idxs])
            # print('alloc_volumes', alloc_volumes)
            # print('scores', scores)
            alloc_delta = curr_volume - torch.sum(alloc_volumes)
            # print('alloc_delta', alloc_delta)
            #print(len(assignable_idxs), alloc_delta, torch.sum(alloc_volumes), curr_volume)
        return alloc_volumes.int()

    def mask_to_target_ratio(self, model, curr_volume, temperature=0.1, eps=1e-8, metric='ipt'): 
        scores = []
        # Calculate the importance score for each sub matrix 
        for name, module in model.named_modules(): 
            if hasattr(module, "In_Qs"): 
                if metric == 'ipt':
                    ipt_in_score = self.calculate_score(f"{name}.In_Qs", p=module.In_Qs, metric="ipt").mean()
                    ipt_out_score = self.calculate_score(f"{name}.Out_Qs", p=module.Out_Qs, metric="ipt").mean()
                    # print(ipt_in_score, ipt_out_score)
                    scores.append(ipt_in_score + ipt_out_score)
                elif metric == 'mag':
                    ipt_in_score = self.calculate_score(f"{name}.In_Qs", p=module.In_Qs, metric="mag").mean()
                    ipt_out_score = self.calculate_score(f"{name}.Out_Qs", p=module.Out_Qs, metric="mag").mean()
                    # print(ipt_in_score, ipt_out_score)
                    scores.append(ipt_in_score + ipt_out_score)
                elif metric == 'w_mag':
                    ipt_score = self.calculate_score(f"{name}", module=module, metric='w_mag').mean()
                    # comb_ipt = torch.mean(ipt_score, dim=1, keepdim=True)
                    scores.append(ipt_score)
        scores = torch.Tensor(scores)
        # print(scores.shape)
        if torch.sum(scores) == 0:
            scores = 1 / scores.shape[0]
        else:
            scores = scores / torch.sum(scores)
        #scores = F.softmax(scores/temperature, dim=-1)
        max_alloc_volumes = torch.Tensor(self.max_alloc_list).squeeze() # max assignable volumes for each layer
        # print(scores)
        alloc_volumes = self.param_alloc(scores, curr_volume, max_alloc_volumes)
        alloc_volume_dict = dict(zip(self.name_list, alloc_volumes.tolist()))
        score_volume_dict = dict(zip(self.name_list, scores.tolist()))
        # change the sparsity ratio for all matrix
        for name, module in model.named_modules(): 
            if hasattr(module, "In_Qs"): 
                updated_thres = 1 - float(alloc_volume_dict[name] / (module.in_features * module.out_features))
                updated_thres = min(max(updated_thres, 0), 1)
                # print(updated_thres, alloc_volume_dict[name])
                # module.init_thres.data.fill_(updated_thres)
                module.init_thres = updated_thres
                alloc_volume_dict[name] = updated_thres

        return alloc_volume_dict, score_volume_dict

    def update(self, model, global_step, metric='ipt'):
        #if global_step<self.total_step-self.final_warmup:
            # Update importance scores element-wise 
            #self.update_ipt(model)
            # do not update ipt during final fine-tuning 
        # Budget schedule
        curr_volume, mask_ind = self.schedule_threshold(global_step)
        if mask_ind:
            # Mask to target budget 
            self.alloc_volume_dict, self.score_volume_dict = self.mask_to_target_ratio(model, curr_volume, metric=metric) 
        return