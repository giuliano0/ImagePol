% Implementação 2D do algoritmo de detecção de re-sampling proposto por
% Alin Popescu e Heny Farid.
clear;

step = 1;
L = 32; % tamanho da imagem aleatória (quadrada por enquanto)
N = 1; % tamanho da vizinhança = |[-N; N]| = 2N+1, (|.| é a cardinalidade do conjunto formado no intervalo)

alpha = rand(2*N+1, 2*N+1);
alpha(1+N, 1+N) = 0;
alpha = alpha / sum(sum(alpha));

alpha_new = zeros(2*N+1, 2*N+1);

% gera imagem aleatória
f = rand(L, L);
f = f - min(min(f));
f = (f / max(max(f))) * 255; % deixar no intervalo [0;1], como no Farid_EM em testes menores

% adiciona correlação aos pixels
% TODO
alpha_0 = [0.12 0.09 0.14; 0.2 0 0.09; 0.2 0.05 0.11];

for x = 4:4:L-4
    for y = 5:6:L-3
        
        f(x, y) = 0;
        
        for u = -1:1
            for v = -1:1
                f(x, y) = f(x, y) + alpha_0(u + 2, v + 2) * f(x + u, y + v);
            end
        end
        
    end
end

max_steps = 1000;

% parâmetros do EM
sigma = 0.05; % variancia da gaussiana
delta = 1/(max(max(f)) - min(min(f)) + 1); % delta uniforme; mais tarde, delta é invertido, representa Pr{f(x,y)|f(x,y) pert. M2}

%pause;

while (step < max_steps)
    % Passo E
    
    % calcula-se a probabilidade de cada valor f(x,y) pertencer ao modelo
    % M1, usando a regra de Bayes. esse passo se resume a calcular a
    % probabilidade para M1 e M2, somá-las, e dividir a prob. de 1 pelo
    % resultado.
    
    r = zeros(L, L);
    w = zeros(L, L);
    P = zeros(L, L);
    
    % Cálculo do residual r
    %for x = 2:L-1
    %    for y = 2:L-1
    %        
    %        r_temp = 0;
    %        
    %        for u = 1:2*N+1
    %            for v = 1:2*N+1
    %                r_temp = r_temp + alpha(u, v) * f(x + u - 2, y + v - 2);
    %            end
    %        end
    %        
    %        r(x, y) = abs(f(x, y) - r_temp); % o abs(.) não é necessário visto que r será elevado ao quadrado, gerando um num. positivo.
    %    end
    %end
    
    filtered_image = filter2(alpha, f);
    r = abs(filtered_image - f);
    clear filtered_image;
    
    for x = 2:L-1
        for y = 2:L-1
            % Cálculo da probabilidade Pr{f(x,y) pert. M1 | f(x, y)}
            % SPEEDUP: colocar sqrt(2*pi) em constante fora do while ou
            % 1/sigma... em constante dentro do while. O mesmo para
            % -1/sigma^2.
            %P(x, y) =  (1 / (sigma * sqrt(2*pi))) * exp( (-(r(x, y)^2) / (2 * sigma^2)) );
            P(x, y) =  exp( (-(r(x, y)^2) / (2 * sigma^2)) );
            
            % Cálculo da probabilidade Pr{f(x,y) | f(x,y) pert. M1}
            w(x, y) = P(x, y) / (P(x, y) + delta);
        end
    end
    
    P(1, :) = [];
    P(L-1, :) = [];
    P(:, 1) = [];
    P(:, L-1) = [];
    
    w(1, :) = [];
    w(L-1, :) = [];
    w(:, 1) = [];
    w(:, L-1) = [];
    
    r(1, :) = [];
    r(L-1, :) = [];
    r(:, 1) = [];
    r(:, L-1) = [];
    
    % Passo M
    for s = 1:(2*N+1)
        for t = 1:(2*N+1)
            A_temp = 0; B_temp = 0; % Variáveis temporárias para A(s,t) e B(s,t)

            for u = 1:(2*N+1)
                for v = 1:(2*N+1)

                    for x = 2:L-2
                        for y = 2:L-2
                            A_temp = A_temp + w(x, y) * f(x + s - 2, y + t - 2) * f(x + u - 2, y + v - 2); % calcula cada coeficiente de alpha(u,v)
                        end
                    end
                    
                    %A_temp = alpha(u, v) * A_temp;
                end
            end
            
            for x = 2:L-2
                for y = 2:L-2
                    B_temp = B_temp + w(x, y) * f(x + s - 2, y + t - 2) * f(x, y); % e o resultado associado
                end
            end
            

            A(s, t) = A_temp;
            B(s, t) = B_temp;
        end
    end
    
    %disp(sprintf('paused'));
    %pause;
    
    % monta e resolve 2*N+1 sistemas lineares
    for n = 1:2*N+1
        % A.alpha(:,n) = B(:,n)
        alpha_new(:,n) = linsolve(A, B(:,n));
    end
    
    alpha_new(1+N, 1+N) = 0;
    
    % condição de parada 
    if (norm(alpha_new - alpha) < 0.01)
        break;
    end

    alpha = alpha_new;
    
    sigma = sqrt(sum(sum(w * r.^2)) / sum(sum(w)));
    
    %disp(sprintf('step %d completed.', step));
    step = step + 1;
    
    %pause(1);
end

F = int8(round(abs(fftshift(fft(P)))));

disp(sprintf('EM finished on step %d. Try imshow(F) to show the Fourier Spectrum.', step));