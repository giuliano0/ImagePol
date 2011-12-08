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

% Normalizando a imagem para [0;1]
% Coloquei esse passo para observar os valores de saída quando a entrada tá
% nesse range. Comentar/retirar se achar necessário.
%f = f - min(min(f));
%f = f / max(max(f));

% constantes de controle
step = 1;
epsilon = 0.01; % condição de parada. Arbitrária
L = length(f); % estamos usando imagens quadradas
N = 2; % tamanho da vizinhança = |[-N; N]| = 2N+1, (|.| é o número de inteiros no intervalo)
neigh = 2*N + 1; % constante para facilitar o cálculo de vizinhança

% inicializa alpha aleatório
alpha = rand(neigh, neigh);
alpha(1 + N, 1 + N) = 0;
alpha = alpha / sum(sum(alpha)); % para garantir que os oeficientes somem 1

% pré-inicializa alpha_new (o matlab recomenda =P)
alpha_new = zeros(neigh, neigh);

max_steps = 10;
converged = false;

% parâmetros do EM
sigma = 0.0075; % variancia da gaussiana fixada via paper
delta = 1/256; % delta uniforme; representa Pr{f(x,y)|f(x,y) pert. M2}

%pause;

while (step < max_steps && ~converged)
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
    
    disp('E step completed');
    pause(0.25);
    
    pause;
    
    % "interlude": padding de f e w, para o passo M
    
    f_size = size(f);
    
    f_padded = [zeros(1, f_size(2)); f; zeros(1, f_size(2))];
    f_padded = ( [zeros(1, f_size(1) + 2); f_padded'; zeros(1, f_size(1) + 2)] )';
    
    clear f_size;
    
    w_size = size(w);
    
    w_padded = [zeros(1, w_size(2)); w; zeros(1, w_size(2))];
    w_padded = ( [zeros(1, w_size(1) + 2); w_padded'; zeros(1, w_size(1) + 2)] )';
    
    clear w_size;
    
    P_img = P - min(min(P));
    P_img = int8(256*(P_img / max(max(P))));
    %imshow(P_img);
    w_img = w - min(min(w));
    w_img = int8(256*(w_img / max(max(w_img))));
    %imshow(w_img);
    
    %disp('Paused before M step.');
    %pause();
    
    % Passo M
    
    A = zeros(neigh^2, neigh^2); B = zeros(neigh^2, 1);
    
    for s = -N:N
        for t = -N:N

            for u = -N:N
                for v = -N:N

                    % Os limites 1+N:L-N são para evitar problemas de
                    % out-f-bounds
                    for x = 1+N:L-N % (1+1 : 128-1)
                        for y = 1+N:L-N % (1+1 : 128-1)
                            i = neigh * (s + N) + (t + (N+1));
                            j = neigh * (u + N) + (v + (N+1));
                            
                            A(i, j) = A(i, j) + w_padded(x, y) * f_padded(x + s, y + t) * f_padded(x + u, y + v);
                        end
                    end
                    
                end
            end
            
            for x = N+1:L-N % (1+1 : 5-1)
                for y = N+1:L-N % (1+1 : 5-1)
                    i = neigh * (s + N) + (t + (N+1));
                    B(i, 1) = B(i, 1) + w_padded(x, y) * f_padded(x + s, y + t) * f_padded(x, y);
                end
            end

        end
    end
    
    % não precisamos mais de i e j
    clear i; clear j;
    
    % era para isso funcionar, eu acho, mas não funcionou, anyway. Depois
    % eu vejo o que deu de errado.
    %A = sum(sum(w)) * A; % multiplica cada termo pela soma dos pesos
    %B = (sum(sum(w)) * sum(sum(f)) * B); % mesmo ;), e B tem q ser vetor-coluna
    
    % aqui, alpha_new (matriz) é linearizada por LINHAS, i.e.:
    % alpha = [a11 a12 a13 a21 a22 a23 a31 a32 a33]'
    % e é chamada de alpha_new_vector.
    
    alpha_new_vector = linsolve(A, B);
    
    % transforma alpha_new_vector em matriz
    for i = 1:neigh
        alpha_new(:, i) = alpha_new_vector(neigh*(i-1) + 1:neigh*i);
    end
    
    % solução noob pra problema sério
    alpha_new(N+1, N+1) = 0;
    
    disp('M step competed');
    pause(0.25);
    
    % condição de parada 
    if (norm(alpha_new - alpha) < epsilon)
        converged = true;
    %else
    %    Rinse, repeat :)
    end

    alpha = alpha_new;
    
    % calcula novo sigma
    % DESABILITEI esse trecho pois o paper diz que foi usado sigma fixo
    % (mesmo que ele tenha exposto o algoritmo com o cálculo, anyway).
    %above = w .* (r.^2);
    %sigma = sqrt(sum(sum(above)) / sum(sum(w)));
    %clear above;
    
    %disp(sprintf('step %d completed.', step));
    fprintf('step %d completed.', step);
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


