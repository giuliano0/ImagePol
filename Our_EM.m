clear;

N = 3; % tamanho da vizinhan�a
L = 10; % tamanho da imagem aleat�ria (quadrada por enquanto)
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
alpha = rand(N, N);
alpha(ceil(N/2), ceil(N/2)) = 0; % o termo do meio da vizinhan�a deve ser 0 (eu acho =P)
sigma = 0.005; % variancia da gaussiana
delta = 10; % delta uniforme

while (1)
    % Passo E

    % calcula o erro e a probabilidade
    for x = 2:L-N
        for y = 2:L-N
            % calcula o h4x_term, que � subtraido em cada itera��o abaixo e
            % portanto vale a pena ser calculado de uma vez
            for u = 1:N
                for v = 1:N
                    h4x_term(x, y) = alpha(u, v) * f(x + u, y + v);
                end
            end
            
            r(x, y) = f(x, y) - h4x_term(x, y);
            temp = exp(-r(x, y)^2 / sigma);
            w(x, y) = temp / (temp + 1 / delta);
            clear temp;
        end
    end
    
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