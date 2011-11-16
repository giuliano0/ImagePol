% Caso trivial de vizinhança 3
clear;

% constantes de controle
step = 1;
L = 3; % tamanho da imagem aleatória (quadrada por enquanto)
N = 1; % tamanho da vizinhança = |[-N; N]| = 2N+1, (|.| é a cardinalidade do conjunto formado no intervalo)

% imagem de caso trivial, 1 pixel relacionado
f = [100 58 200; 90 146 219; 31 160 252];

% matriz de correlação dos pixels
alpha_0 = [0.25 0 0.25; 0 0 0; 0.25 0 0.25];

% inicializa alpha aleatório
alpha = rand(3, 3);
alpha(2, 2) = 0;
alpha = alpha / sum(sum(alpha)); % para garantir que os oeficientes somam 1

% pré-inicializa alpha_new (o matlab recomenda =P)
alpha_new = zeros(2*N+1, 2*N+1);

max_steps = 10;

% parâmetros do EM
sigma = 20; % variancia da gaussiana
delta = 1/256; % delta uniforme; mais tarde, delta é invertido, representa Pr{f(x,y)|f(x,y) pert. M2}

%pause;

while (step < max_steps)
    % Passo E

    %r = zeros(L, L);
    w = zeros(L, L);
    P = zeros(L, L);
    
    filtered_image = filter2(alpha, f);
    r = abs(filtered_image - f);
    clear filtered_image;
    
    % Gera P e w SEM PADDING
    for x = 1:3
        for y = 1:3
            % Cálculo da probabilidade Pr{f(x,y) pert. M1 | f(x, y)}
            P(x, y) =  (1 / (sigma * sqrt(2*pi))) * exp( -((r(x, y)^2) / (2 * sigma^2)) );
            %P(x, y) =  exp( -(r(x, y)^2 / (2 * sigma^2)) );
            
            % Cálculo da probabilidade Pr{f(x,y) | f(x,y) pert. M1}
            w(x, y) = P(x, y) / (P(x, y) + delta);
        end
    end
    
    disp('E step completed');
    pause(0.25);
    
    % "interlude": padding de f e w, para o passo M
    
    f_size = size(f);
    
    f_padded = [zeros(1, f_size(2)); f; zeros(1, f_size(2))];
    f_padded = ( [zeros(1, f_size(1) + 2); f_padded'; zeros(1, f_size(1) + 2)] )';
    
    clear f_size;
    
    w_size = size(w);
    
    w_padded = [zeros(1, w_size(2)); w; zeros(1, w_size(2))];
    w_padded = ( [zeros(1, w_size(1) + 2); w_padded'; zeros(1, w_size(1) + 2)] )';
    
    clear w_size;
    
    % Passo M
    
    A = zeros(9, 9); B = zeros(9, 1);
    
    for s = -1:1
        for t = -1:1

            for u = -1:1
                for v = -1:1

                    for x = 2:4 % (1+1 : 5-1)
                        for y = 2:4 % (1+1 : 5-1)
                            i = 3 * (s + 1) + (t + 2);
                            j = 3 * (u + 1) + (v + 2);
                            
                            A(i, j) = A(i, j) + w_padded(x, y) * f_padded(x + s, y + t) * f_padded(x + u, y + v);
                        end
                    end
                    
                end
            end
            
            for x = 2:4 % (1+1 : 5-1)
                for y = 2:4 % (1+1 : 5-1)
                    B(3*(s+1) + t+2, 1) = B(3*(s+1) + t+2, 1) + w_padded(x, y) * f_padded(x + s, y + t) * f_padded(x, y);
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
    alpha_new(:, 1) = alpha_new_vector(1:3);
    alpha_new(:, 2) = alpha_new_vector(4:6);
    alpha_new(:, 3) = alpha_new_vector(7:9);
    
    % solução noob pra problema sério
    alpha_new(2, 2) = 0;
    
    disp('M step competed');
    pause(0.25);
    
    % condição de parada 
    if (norm(alpha_new - alpha) < 0.01)
        break;
    end

    alpha = alpha_new;
    
    above = 0;
    
    for x = 1:3
        for y = 1:3
            above = above + w(x,y) * r(x,y)^2;
        end
    end
    
    sigma = sqrt(above / sum(sum(w)));
    clear above;
    
    disp(sprintf('step %d completed.', step));
    step = step + 1;
    
    pause(0.5);
end

F = int8(round(abs(fftshift(fft(P)))));

disp(sprintf('EM finished on step %d. Try imshow(F) to show the Fourier Spectrum.', step));
