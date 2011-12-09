% ASSUMIR RESAMPLING POR INTERPOLAÇÃO PORCARIA (digo, linear)!
% nvm, interpolação bicúbica foi assumida no paper.
% - De acordo com os autores, interpolações não lineares também produzem
% relações, mas elas podem não ser (tão) periódicas.
% - O blurring (a filtragem mostrada no algoritmo) não é necessária, mas é
% reportado pelos autores que acelera a convergência.

% Os parâmetros do EM foram fixados a:
% - N = 2 (tamanho de vizinhança 5x5)
% - Nh = 3 (tamanho do filtro de blurring 3x3)
% - sigma_0 = 0.0075
% - p0 = delta = 1/256.

clear;

f = double(imread('sample_10dg.png'));
%load('f_test.mat');

% Normalizando a imagem para [0;1]
% Coloquei esse passo para observar os valores de saída quando a entrada tá
% nesse range. Comentar/retirar se achar necessário.
%f = f - min(min(f));
%f = f / max(max(f));

% constantes de controle
step = 1;
max_steps = 20;
converged = false;
epsilon = 0.1; % condição de parada. Arbitrária
L = length(f); % estamos usando imagens quadradas
N = 2; % tamanho da vizinhança = |[-N; N]| = 2N+1, (|.| é o número de inteiros no intervalo)
neigh = 2*N + 1; % constante para facilitar o cálculo de vizinhança
% DEBUG
least_normdiff = 100; % guarda a menor norma encontrada no processo
% END DEBUG

% inicializa alpha aleatório
alpha = rand(neigh, neigh);
alpha(1 + N, 1 + N) = 0;
alpha = alpha / sum(sum(alpha)); % para garantir que os coeficientes somem 1

% pré-inicializa alpha_new (o matlab recomenda =P)
alpha_new = zeros(neigh, neigh);

% parâmetros do EM
sigma = 1.0; % variancia da gaussiana fixada via paper
delta = 1/256; % delta uniforme; representa Pr{f(x,y)|f(x,y) pert. M2}

%pause;

while (step <= max_steps && ~converged)
    % Passo E
    % Estima a probabilidade de cada sample f(x,y) pertencer a um modelo
    % específico, M1 ou M2, correlacionado ou não, respectivamente.

    filtered_image = filter2(alpha, f);
    r = abs(filtered_image - f);
    %clear filtered_image;
    
    % Gera P e w SEM PADDING
    % Probabilidade Pr{f(x,y) pert. M1 | f(x, y)}
    P =  exp( -((r.^2) / (2 * sigma^2)) ) / (sigma * sqrt(2*pi));
    % Probabilidade Pr{f(x,y) | f(x,y) pert. M1}
     w = P ./ (P + delta);
    
    %disp('E step completed');
    %pause(0.25);
    
    % "interlude": padding de f e w, para o passo M
    z = zeros(length(f), N);
    pf = cat(2, z, f, z);
    z = zeros(length(f) + 2*N, N);
    pf = cat(2, z, pf', z);
    
    z = zeros(length(w), N);
    pw = cat(2, z, w, z);
    z = zeros(length(w) + 2*N, N);
    pw = cat(2, z, pw', z);
    
    clear z;
    
    % DEBUG: calcula IBAGENS de P e W se alguém quiser vê-las.
    P_img = uint8(255*P);
    %P_img = P - min(min(P));
    %P_img = uint8(255*(P_img / max(max(P))));
    name = sprintf('P_and_w_images/P%d.png', step);
    imwrite(P_img, name);
    
    w_img = uint8(255*w);
    %w_img = w - min(min(w));
    %w_img = uint8(255*(w_img / max(max(w_img))));
    name = sprintf('P_and_w_images/w%d.png', step);
    imwrite(w_img, name);
    
    %disp('Paused before M step.');
    %pause();
    %END DEBUG
    
    % Passo M
    
    A = zeros(neigh, neigh); B = zeros(neigh, neigh);
    
    for s = -N:N
        for t = -N:N
            i = 1+ s + N; j = 1 + t + N;
            aux_sum = 0; % cada termo alpha_uv * somatório em x,y.

            for u = -N:N
                for v = -N:N

                    % Os limites 1+N:L-N são para evitar problemas de
                    % out-f-bounds. Como estamos usando padding em f e w,
                    % o f(1,1) passa a ser f(1+N,1+N)
                    for x = 1+N:L-N
                        for y = 1+N:L-N
                            aux_sum = aux_sum + pw(x, y) * pf(x + s, y + t) * pf(x + u, y + v);
                        end % y
                    end % x
                    
                    % Como requerido no paper, multiplicamos cada soma por
                    % alpha_uv antes de calcular o próximo termo e somar ao
                    % total.
                    A(i, j) = A(i, j) + alpha(1 + u + N, 1 + v + N) * aux_sum;
                    
                    % Feito isso, é calculado o próximo termo (aux_sum),
                    % que é multiplicado por alpha_uv e somado a A(i, j).

                end % v
            end % u
            
            % Sobre limites: ler comentário no loop x,y acima
            for x = 1+N:L-N
                for y = 1+N:L-N
                    B(i, j) = B(i, j) + pw(x, y) * pf(x + s, y + t) * pf(x, y);
                end % y
            end % x

        end % t
    end % s
    
    % não precisamos mais de i e j
    clear i; clear j;
    
    alpha_new = linsolve(A, B);
    
    %fprintf('alpha_new mid coefficient value was %d. Now it is being set to 0.\n', alpha_new(N+1, N+1));
    
    % solução noob pra problema sério
    alpha_new(N+1, N+1) = 0;
    
    %disp('M step competed');
    %pause(0.25);
    
    % condição de parada
    normdiff = norm(alpha_new - alpha);
    
    % DEBUG
    if (normdiff < least_normdiff)
        least_normdiff = normdiff;
    end
    % END DEBUG
    
    if (normdiff < epsilon)
        converged = true;
    %else
    %    Rinse, repeat :)
    end

    alpha = alpha_new;
    
    % calcula novo sigma
    % DESABILITEI esse trecho pois o paper diz que foi usado sigma fixo
    % (mesmo que ele tenha exposto o algoritmo com o cálculo, anyway).
    aux = w .* (r.^2);
    sigma = sqrt(sum(sum(aux)) / sum(sum(w)));
    clear aux;
    
    fprintf('sigma = %d\n', sigma);
    
    %disp(sprintf('step %d completed.', step));
    %fprintf('step %d completed. Norm of diference between alphas is %d\n', step, normdiff);
    step = step + 1;
    
    %pause(0.5);
end

if (converged)
    F = abs(fftshift(fft(P)));
    F = F - min(min(F));
    F = 255*(F / max(max(F)));
    F = int8(F);

    %disp(sprintf('EM finished on step %d. Try imshow(F) to show the Fourier Spectrum.', step));
    fprintf('EM finished on step %d. Try imshow(F) to show the Fourier Spectrum.', step);
else
    disp('EM finished without converging. :(');
end

FP = abs(fftshift(fft(P)));
FP = FP - min(min(FP));
FP = 255*(FP / max(max(FP)));
FP = int8(FP);

FW = abs(fftshift(fft(w)));
FW = FW - min(min(FW));
FW = 255*(FW / max(max(FW)));
FW = int8(FW);

w2 = w - min(min(w));
w2 = 255*(w2 / max(max(w2)));
w2 = int8(w2);

p2 = w - min(min(P));
p2 = 255*(p2 / max(max(p2)));
p2 = int8(p2);

