% ASSUMIR RESAMPLING POR INTERPOLA��O PORCARIA (digo, linear)!
% nvm, interpola��o bic�bica foi assumida no paper.
% - De acordo com os autores, interpola��es n�o lineares tamb�m produzem
% rela��es, mas elas podem n�o ser (t�o) peri�dicas.
% - O blurring (a filtragem mostrada no algoritmo) n�o � necess�ria, mas �
% reportado pelos autores que acelera a converg�ncia.

% Os par�metros do EM foram fixados a:
% - N = 2 (tamanho de vizinhan�a 5x5)
% - Nh = 3 (tamanho do filtro de blurring 3x3)
% - sigma_0 = 0.0075
% - p0 = delta = 1/256.

clear;

f = double(imread('sample_10dg.png'));

% Normalizando a imagem para [0;1]
% Coloquei esse passo para observar os valores de sa�da quando a entrada t�
% nesse range. Comentar/retirar se achar necess�rio.
%f = f - min(min(f));
%f = f / max(max(f));

% constantes de controle
step = 1;
epsilon = 0.01; % condi��o de parada. Arbitr�ria
L = length(f); % estamos usando imagens quadradas
N = 2; % tamanho da vizinhan�a = |[-N; N]| = 2N+1, (|.| � o n�mero de inteiros no intervalo)
neigh = 2*N + 1; % constante para facilitar o c�lculo de vizinhan�a

% inicializa alpha aleat�rio
alpha = rand(neigh, neigh);
alpha(1 + N, 1 + N) = 0;
alpha = alpha / sum(sum(alpha)); % para garantir que os oeficientes somem 1

% pr�-inicializa alpha_new (o matlab recomenda =P)
alpha_new = zeros(neigh, neigh);

max_steps = 10;
converged = false;

% par�metros do EM
sigma = 0.0075; % variancia da gaussiana fixada via paper
delta = 1/256; % delta uniforme; representa Pr{f(x,y)|f(x,y) pert. M2}

%pause;

while (step < max_steps && ~converged)
    % Passo E
    % Estima a probabilidade de cada sample f(x,y) pertencer a um modelo
    % espec�fico, M1 ou M2, correlacionado ou n�o, respectivamente.

    filtered_image = filter2(alpha, f);
    r = abs(filtered_image - f);
    %clear filtered_image;
    
    % Gera P e w SEM PADDING
    % Probabilidade Pr{f(x,y) pert. M1 | f(x, y)}
    P =  exp( -((r.^2) / (2 * sigma^2)) ) / (sigma * sqrt(2*pi));
    % Probabilidade Pr{f(x,y) | f(x,y) pert. M1}
    w = P ./ (P + delta);
    
    disp('E step completed');
    pause(0.25);
    
    % "interlude": padding de f e w, para o passo M
    z = zeros(length(f), N);
    pf = cat(2, z, f, z);
    z = zeros(length(f) + 2*N, N);
    pf = cat(2, z, pf', z);
    
    z = zeros(length(w), N);
    pw = cat(2, z, w, z);
    z = zeros(length(w) + 2*N, N);
    pw = cat(2, z, pw', z);
    
    % DEBUG: calcula IBAGENS de P e W se algu�m quiser v�-las.
    %P_img = P - min(min(P));
    %P_img = int8(256*(P_img / max(max(P))));
    %imshow(P_img);
    
    %w_img = w - min(min(w));
    %w_img = int8(256*(w_img / max(max(w_img))));
    %imshow(w_img);
    
    %disp('Paused before M step.');
    %pause();
    %END DEBUG
    
    % Passo M
    
    A = zeros(neigh, neigh); B = zeros(neigh, neigh);
    
    for s = -N:N
        for t = -N:N
            i = 1+ s + N; j = 1 + t + N;
            aux_sum = 0; % cada termo alpha_uv * somat�rio em x,y.

            for u = -N:N
                for v = -N:N

                    % Os limites 1+N:L-N s�o para evitar problemas de
                    % out-f-bounds. Como estamos usando padding em f e w,
                    % o f(1,1) passa a ser f(1+N,1+N)
                    for x = 1+N:L-N
                        for y = 1+N:L-N
                            aux_sum = aux_sum + pw(x, y) * pf(x + s, y + t) * pf(x + u, y + v);
                        end % y
                    end % x
                    
                    % Como requerido no paper, multiplicamos cada soma por
                    % alpha_uv antes de calcular o pr�ximo termo e somar ao
                    % total.
                    A(i, j) = A(i, j) + alpha(1 + u + N, 1 + v + N) * aux_sum;
                    
                    % Feito isso, � calculado o pr�ximo termo (aux_sum),
                    % que � multiplicado por alpha_uv e somado a A(i, j).

                end % v
            end % u
            
            % Sobre limites: ler coment�rio no loop x,y acima
            for x = 1+N:L-N
                for y = 1+N:L-N
                    B(i, j) = B(i, j) + pw(x, y) * pf(x + s, y + t) * pf(x, y);
                end % y
            end % x

        end % t
    end % s
    
    % n�o precisamos mais de i e j
    clear i; clear j;
    
    alpha_new_vector = linsolve(A, B);
    
    % transforma alpha_new_vector em matriz
    for i = 1:neigh
        alpha_new(:, i) = alpha_new_vector(neigh*(i-1) + 1:neigh*i);
    end
    
    % solu��o noob pra problema s�rio
    alpha_new(N+1, N+1) = 0;
    
    disp('M step competed');
    pause(0.25);
    
    % condi��o de parada 
    if (norm(alpha_new - alpha) < epsilon)
        converged = true;
    %else
    %    Rinse, repeat :)
    end

    alpha = alpha_new;
    
    % calcula novo sigma
    % DESABILITEI esse trecho pois o paper diz que foi usado sigma fixo
    % (mesmo que ele tenha exposto o algoritmo com o c�lculo, anyway).
    %above = w .* (r.^2);
    %sigma = sqrt(sum(sum(above)) / sum(sum(w)));
    %clear above;
    
    %disp(sprintf('step %d completed.', step));
    fprintf('step %d completed.\n', step);
    step = step + 1;
    
    pause(0.5);
end

if (converged)
    F = int8(abs(fftshift(fft(P))));

    %disp(sprintf('EM finished on step %d. Try imshow(F) to show the Fourier Spectrum.', step));
    fprintf('EM finished on step %d. Try imshow(F) to show the Fourier Spectrum.', step);
else
    disp('EM finished without converging. :(');
end


