%% This file is to generate the matrix for generating colored noise.

clear;

eta = 0.8;
N = 576;
cov = zeros(N, N);
for ii=1:N
    for jj=ii:N
        cov(ii,jj) = eta^(abs(ii-jj));
        cov(jj,ii) = cov(ii,jj);
    end
end
transfer_mat = cov^(1/2);
fout = fopen(sprintf('cov_1_2_corr_para%.2f.dat', eta),'wb');
fwrite(fout, transfer_mat, 'single');
fclose(fout);
