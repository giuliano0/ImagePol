clear;

step = 1;
L = 10; % tamanho da imagem aleatória (quadrada por enquanto)
N = 1; % tamanho da vizinhança = |[-N; N]| = 2N+1, (|.| é a cardinalidade do conjunto formado no intervalo)
alpha = rand(2*N+1, 2*N+1);
alpha(ceil(N/2), ceil(N/2)) = 0; % o termo do meio da vizinhança deve ser 0 (eu acho =P)
alpha_new = zeros(2*N+1, 2*N+1);
w = zeros(L, L);
P = zeros(L, L);

% gera imagem aleatória
f = rand(L, L);
f = f - min(min(f));
f = (f / max(max(f)));% * 255;

% adiciona correlação aos pixels
% TODO

% parâmetros do EM
sigma = 0.5; % variancia da gaussiana
delta = 1/(max(max(f)) - min(min(f))); % delta uniforme; mais tarde, delta é invertido, representa Pr{f(x,y)|f(x,y) pert. M2}

while (1)
    % Passo E
    
    % calcula-se a probabilidade de cada valor f(x,y) pertencer ao modelo
    % M1, usando a regra de Bayes. esse passo se resume a calcular a
    % probabilidade para M1 e M2, somá-las, e dividir a prob. de 1 pelo
    % resultado.
    
    r = zeros(L, L);
    
    % Cálculo do residual r
    for x = 2:L-1
        for y = 2:L-1
            
            r_temp = 0;
            
            for u = 1:2*N+1
                for v = 1:2*N+1
                    r_temp = r_temp + alpha(u, v) * f(x + u - 2, y + v - 2);
                end
            end
           
            r(x, y) = abs(f(x, y) - r_temp);
        end
    end
    
    for x = 2:L-1
        for y = 2:L-1
            % Cálculo da probabilidade Pr{f(x,y) pert. M1 | f(x, y)}
            % SPEEDUP: colocar sqrt(2*pi) em constante fora do while ou
            % 1/sigma... em constante dentro do while. O mesmo para
            % -1/sigma^2.
            P(x, y) =  (1 / (sigma * sqrt(2*pi))) * exp( (-(r(x, y)^2) / (2 * sigma^2)) );
            
            % Cálculo da probabilidade Pr{f(x,y) | f(x,y) pert. M1}
            w(x, y) = P(x, y) / (P(x, y) + delta);
        end
    end
    
    % Passo M
    for s = 1:(2*N+1)
        for t = 1:(2*N+1)
            A_temp = 0; B_temp = 0;

            for u = 1:(2*N+1)
                for v = 1:(2*N+1)

                    for x = 2:L-2
                        for y = 2:L-2
                            A_temp = A_temp + w(x, y) * f(x+s-2, y+t-2) * f(x+u-2, y+v-2); % calcula cada coeficiente de alpha(u,v)
                            B_temp = B_temp + w(x, y) * f(x+s-2, y+t-2) * f(x, y); % e o resultado associado
                        end
                    end
                    
                    A_temp = alpha(u, v) * A_temp;
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
    
    alpha_new(ceil(N/2), ceil(N/2)) = 0;
    
    % condição de parada 
    if (norm(alpha_new - alpha) < 0.1)
        break;
    end

    alpha = alpha_new;
    
    sigma = sqrt(sum(sum(w * r.^2)) / sum(sum(w)));
    
    disp(sprintf('step %d completed.', step));
    step = step + 1;
    
    pause(1);
end