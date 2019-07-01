clear all
clc
%parameter-----------------------------------------------------------------
z = 54;        %lift number
inputtype = 'loc_row_nonzero'; %'loc_row_nonzero','P','Hlist'
getP = 'False';
%load data-----------------------------------------------------------------
if strcmp(inputtype, 'loc_row_nonzero')
    loc_row_nonzero = textread('C:\Users\user\Desktop\loc_row_nonzero.txt');
elseif strcmp(inputtype, 'P')
    protograph = textread('F:\decoding optimaztion work\SRC_23to34_p_54.txt');
elseif strcmp(inputtype, 'Hlist')
    Hlist = textread('F:\decoding optimaztion work\LDPC decode\LDPC_matrix\LDPC_chk_mat_1152_864.txt');
end

%start---------------------------------------------------------------------
if strcmp(inputtype, 'loc_row_nonzero')||strcmp(inputtype, 'Hlist')
    if strcmp(inputtype, 'loc_row_nonzero')
        H_index = [];
        [n,m] = size(loc_row_nonzero);
        for i = 1:n
            H_index = [H_index;[i*ones(length(find(loc_row_nonzero(i,:)~=0)),1),loc_row_nonzero(i,find(loc_row_nonzero(i,:)~=0))']];
        end
        H_index = H_index-1;
        H = [];
        for i  = 1:length(H_index(:,1))
        H(H_index(i,1)+1,H_index(i,2)+1) = 1;
        end
    elseif strcmp(inputtype, 'Hlist')
        for i  = 1:length(Hlist(:,1))
            H(Hlist(i,1)+1,Hlist(i,2)+1) = 1;
        end
    end
    [G,Glist,H,Hlist]= GaussianXY(H);
    [K,N] = size(G); M = N - K;
    %get file name
    G_file = strcat('./LDPC_gen_mat_',num2str(N),'_',num2str(K),'.txt');
    H_file = strcat('./LDPC_chk_mat_',num2str(N),'_',num2str(K),'.txt');
    %save G_matrix
    fid1 = fopen(G_file,'w');
    fprintf(fid1,'%d\t%d\r\n',Glist);
    fclose(fid1);
    %save H_matrix
    fid2 = fopen(H_file,'w');
    fprintf(fid2,'%d\t%d\r\n',Hlist);
    fclose(fid2);
    
    if strcmp(getP, 'True')
        Mp = M/z; Np = N/z;
        protograph = zeros(Mp,Np);
        for m = 1:Mp
            for n = 1:Np
                if H((z*m-z+1):(z*m),(z*n-z+1):(z*n)) == zeros(z,z)
                    protograph(m,n) = -1;
                elseif sum(H((z*m-z+1),(z*n-z+1):(z*n))) == 1
                    shift = min(find(H((z*m-z+1),(z*n-z+1):(z*n))))-1;
                    if H((z*m-z+1):(z*m),(z*n-z+1):(z*n)) == circshift(eye(z,z),shift,2)
                        protograph(m,n) = shift;
                    else
                        protograph(m,n) = -2;
                    end
                else
                    protograph(m,n) = -2;
                end
            end
        end
        [px,py]=find(protograph' > -2);
        Plist = [py-1,px-1,reshape(protograph',[Mp*Np,1])]';
        P_file = strcat('./LDPC_pro_mat_',num2str(N),'_',num2str(K),'.txt');
        fid = fopen(P_file,'w');
        fprintf(fid,'%d\t%d\t%d\r\n',Plist);
        fclose(fid);
    end
end
%start---------------------------------------------------------------------
if strcmp(inputtype, 'P')
    l = find((protograph + 1)~=0);
    ln = ceil(l/24);lm = l-24*(ceil(l/24)-1);
    for i = 1:length(l)
        protograph(l) = mod(protograph(l), 24);
    end
    [Mp,Np] = size(protograph);
    M = Mp*z; N = Np*z;K =N-M;
    H = zeros(M,N);
    for m = 1:Mp
        for n = 1:Np
            if protograph(m,n) == -1
                H((z*m-z+1):(z*m),(z*n-z+1):(z*n)) = zeros(z,z);
            else
                H((z*m-z+1):(z*m),(z*n-z+1):(z*n)) = circshift(eye(z,z),protograph(m,n),2);
            end
        end
    end
    [G,Glist,H,Hlist]= GaussianXY(H);
    [a,b]=find(protograph' > -2);
    Plist = [b-1,a-1,reshape(protograph',m*n,1)]';
    %get file name
    G_file = strcat('./LDPC_gen_mat_',num2str(N),'_',num2str(K),'.txt');
    H_file = strcat('./LDPC_chk_mat_',num2str(N),'_',num2str(K),'.txt');
    P_file = strcat('./LDPC_pro_mat_',num2str(N),'_',num2str(K),'.txt');
    %save G_matrix
    fid1 = fopen(G_file,'w');
    fprintf(fid1,'%d\t%d\r\n',Glist);
    fclose(fid1);
    %save H_matrix
    fid2 = fopen(H_file,'w');
    fprintf(fid2,'%d\t%d\r\n',Hlist);
    fclose(fid2);
    %save P_matrix
    fid3 = fopen(P_file,'w');
    fprintf(fid3,'%d\t%d\t%d\r\n',Plist);
    fclose(fid3);
end

%customized function-------------------------------------------------------
function [G, Glist,H,Hlist] = GaussianXY(input)
    H = input;
    H1 = H;
    H0 = H;
    [M,N] = size(H);
    K = N - M;
    I = diag(ones(1,N));
    for i = 1:M
        if H1(i,i+K)==0
            j = max(find(H1(i,:)));
            H1(:,[i+K,j]) = H1(:,[j,i+K]);
            H0(:,[i+K,j]) = H0(:,[j,i+K]);
            I(:,[i+K,j]) = I(:,[j,i+K]);
            for k = i+1:M
                if H1(k,i+K) == 1
                    H1(k,:) = H1(i,:)+H1(k,:);
                    H1(k,:) = mod(H1(k,:), 2);
                end
            end
        else
            for k = i+1:M
                if H1(k,i+K) == 1
                    H1(k, :) = H1(i,:) +H1(k,:);
                    H1(k, :) = mod(H1(k, :),2);
                end
            end
        end
    end
    
    for i = M:-1:2
        for k = i-1:-1:1
            if H1(k,i+K) == 1
                 H1(k,:) = H1(i,:)+H1(k,:);
                 H1(k,:) = mod(H1(k,:), 2);
            end
        end
    end
    PP = H1(:,1:K);
    G = [diag(ones(1,K)) PP']*I';
    [a,b]=find(G'==1);
    Glist = [b-1,a-1]';
    [c,d]=find(H'==1);
    Hlist = [d-1,c-1]';
end