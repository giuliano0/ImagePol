clear;

L = 10; % tamanho da imagem aleat�ria (quadrada por enquanto)
N = 1; % tamanho da vizinhan�a = |[-N; N]| = 2N+1, (|.| � a cardinalidade do conjunto formado no intervalo)
alpha = rand(N, N);
alpha(ceil(N/2), ceil(N/2)) = 0; % o termo do meio da vizinhan�a deve ser 0 (eu acho =P)
alpha_new = [];
r = [];
w = [];
W = zeros(L^2, L^2);
E = [];
h4x_term = [];

% gera imagem aleat�ria
f = rand(L, L);
f = f - min(min(f));
f = 255 * (f / max(max(f)));

% adiciona correla��o aos pixels
% TODO

% par�metros do EM
sigma = 0.005; % variancia da gaussiana
delta = 1/(max(max(f)) - min(min(f))); % delta uniforme; mais tarde, delta � invertido, representa Pr{f(x,y)|f(x,y) pert. M2}

while (1)
    % Passo E
    
    % calcula-se a probabilidade de cada valor f(x,y) pertencer ao modelo
    % M1, usando a regra de Bayes. esse passo se resume a calcular a
    % probabilidade para M1 e M2, som�-las, e dividir a prob. de 1 pelo
    % resultado.
    
    for x = 1+N:L-N
        for y = 1+N:L-N
            % C�lculo do residual r
            % Primeiro, calculamos o somat�rio interno
            for u = -N:N
                for v = -N:N
                    r(x, y) = alpha(u, v) * f(x + u, y + v);
                end
            end
           
            % e ent�o subtra�mos essa soma de cada f(x,y)
            r(x, y) = f(x, y) - r(x, y);
            % Fim do c�lculo do residual
            
            % C�lculo da probabilidade Pr{f(x,y) pert. M1 | f(x, y)}
            % SPEEDUP: colocar sqrt(2*pi) em constante fora do while ou
            % 1/sigma... em constante dentro do while. O mesmo para
            % -1/sigma^2.
            P(x, y) =  (1 / (sigma * sqrt(2*pi))) * ((-1 / (2 * sigma^2)) * r(x, y)^2);
            % Fim do c�lculo da probabilidade Pr{f(x,y) pert. M1 | f(x, y)}
            
            % C�lculo da probabilidade Pr{f(x,y) | f(x,y) pert. M1}
            w(x, y) = P(x, y) / (P(x, y) + 1/delta);
            % Fim do c�lculo da probabilidade Pr{f(x,y) | f(x,y) pert. M1}
        end
    end
    
    % Passo M again
    % TODO
    
    % Passo M
    M = [f(1:L-2, 1:L-2)' f(3:L, 3:L)'];
    b = f(2:L-1, 2:L-1)';
    r = r(2:end, 2:end); % remove edge point;
    w = w(2:end, 2:end); % remove edge point
    W = diag(w);
    alpha_new = inv(M'*W*M)*M'*W*b; % WLS
    
    % condi��o de parada 
    if (norm(alpha_new - alpha) < 0.1)
        break;
    end

    alpha = alpha_new;
    clear alpha_new;
    
    sigma = sum(w.*(r.^2)) / sum(w);
end