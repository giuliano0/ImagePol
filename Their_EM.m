function alpha = Their_EM(im)
    A = imread(im);
    A = double(A);
    [~, ~, rgb] = size(A);

    err=2;
    p0 = 1 / (max(A(:))-min(A(:))); 
    alpha = cell(3,1);

    neigh_i = {-1, -1, -1,  0, +1, +1, +1,  0};
    neigh_j = {-1,  0, +1, +1, +1,  0, -1, -1};
    n_neigh = length(neigh_i);

    for c=1:rgb
        s = 1000;
        B = A(:, :, c);
        a = rand([1 8]);
        a_prev = a*10;  % Just to enter in the while loop

        shiftedB = {};
        for l = 1:n_neigh
           shiftedB = cat (2, shiftedB, shiftN(B(:, :), neigh_i{l}, neigh_j{l}));
        end

        while (norm(a - a_prev) > err)
            a_prev = a;
            % Expectation
            acc = 0;
            for k = 1:n_neigh
                acc = acc+a(k) * shiftedB{k};
            end

           r = abs(B - acc);

            r2 = r.^2;
            %exp(-r2/(2*s^2))
            %(s*sqrt(2*pi))
            P = exp(-r2/(2*s^2))/(s*sqrt(2*pi));
            w = P./(P+p0);
            %w = exp(-r2/s)/(exp(-r2/s) + p0);
            % Maximization
            C = zeros(n_neigh,1);
            M = zeros(n_neigh);        
            for k = 1:n_neigh
                aux = w.*shiftedB{k}.*B;
                C(k) = sum(aux(:));
                for l = 1:n_neigh
                    aux = w.*shiftedB{k}.*shiftedB{l};
                    M(k,l) = sum(aux(:));
                end
            end
            a=(M\C)';
            aux=w.*r2;
            s=sqrt(sum(aux)/sum(w));
        end
        alpha{c}=a;    
    end

    function N = shiftN(M,dx,dy)
        [m,n,rgb] = size(M);
        N = zeros(size(M));

        for k = 1:rgb
            MM = zeros(m + 2 * abs(dx), n + 2 * abs(dy));
            MM(abs(dx) + 1:abs(dx) + m, abs(dy) + 1:n + abs(dy)) = M(:, :, k);
            N(:, :, k) = MM(abs(dx) + 1 + dx:abs(dx) + m + dx, abs(dy) + 1 + dy:abs(dy) + n + dy);
        end

        