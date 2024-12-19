function [full_label,sample_lbl,sample_lbl_index]=noise_samples(sample_lbl,asp,method,full_label,sample_lbl_index)
    [~,noise_lbl,noise_index]=Random_sampling(sample_lbl,asp,"all");
    h=unique(sample_lbl);
    noiselbl=[];
    for g=1:length(noise_index)
        for j=h'
            l=setdiff(h,j);  
            if noise_lbl(g)==j
                noiselbl(g)=l(randperm(numel(l),1));
            end
        end
    end
    for u=1:length(noise_index)
        full_label(sample_lbl_index(noise_index(u)))=noiselbl(u);
    end
    
end