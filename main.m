%main function for data fitting
function main()

    %prompt user for  filename and read the data from file
    filename = input('Enter the filename: ', 's');
    data = load(filename);
    
    %extract  X and Y columns from the loaded data
    x = data(:, 1);
    y = data(:, 2);

    %display fitting options
    fprintf('Select the function to fit your data:\n');
    fprintf('1. Polynomial: y = a_0 + a_1*x + ... + a_m*x^m\n');
    fprintf('2. Exponential: y = a*e^(bx)\n');
    fprintf('3. Saturation: y = (ax) / (b + x)\n');
    
    %takes user's choice 
    choice = input('Your selection: ');

    %based on user's choice, fit data and get formula
    switch choice
        case 1
            %prompt for polynomial degree
            degree = input('Enter the degree of the polynomial (1, 2, or 3): ');
    
            %fit polynomial
            [p, R2] = polynomial_fit(x, y, degree);
    
            %constructing formula string for polynomial
            formula = 'y = ';
            for i = 0:degree
                if i == degree && p(degree+1-i) < 0 %handling negative sign for last term
                    formula = [formula, sprintf('%0.4f x^%0.4f', p(degree+1-i), i)];
                else
                    formula = [formula, sprintf('%0.4f x^%0.4f + ', p(degree+1-i), i)];
                end
            end
            formula = formula(1:end-2); %ensures it ends properly without ' + '

        case 2
            %for exponential fit
            [a, b, R2] = exponential_fit(x, y);
            formula = sprintf('y = %0.4f e^%0.4f x', a, b);
            
        case 3
            %for saturation fit
            [a, b, R2] = saturation_fit(x, y);
            formula = sprintf('y = (%0.4f x) / (%0.4f + x)', a, b);
        otherwise
            %handles any invalid choices
            error('Invalid choice.');
    end

    %displays the derived formula and R^2 value
    fprintf('Formula: %s, R^2 = %f\n', formula, R2);

    %plotting raw data
    figure;
    scatter(x, y, 'o');
    hold on;
    
    %depending on the user's choice, compute the values for the fitted curve
    if choice == 1
        xfit = linspace(min(x), max(x), 1000);
        yfit = polyval(p, xfit);
    elseif choice == 2
        xfit = linspace(min(x), max(x), 1000);
        yfit = a * exp(b * xfit);
    else
        xfit = linspace(min(x), max(x), 1000);
        yfit = (a .* xfit) ./ (b + xfit);
    end
    
    %plots the fitted curve
    plot(xfit, yfit, 'r');
    xlabel('X'); ylabel('Y');
    title('Data and Fitted Curve');
    legend('Raw Data', 'Fitted Curve');
    
    %display the formula and the R^2 value on the plot
    text(min(xfit), max(yfit) - 0.1*(max(yfit)-min(yfit)), {formula, ['R^2 = ', num2str(R2)]});
    
    hold off;
end


function [p, R2] = polynomial_fit(x, y, degree)
    %create the matrix A
    A = zeros(length(x), degree+1);
    for i = 0:degree
        A(:, degree+1-i) = x.^i;
    end
    
    %compute the polynomial coefficients
    p = (A' * A) \ (A' * y);
  
    %calculate predicted values using the polynomial
    y_pred = A * p;
    
    %compute R^2
    SSE = sum((y - y_pred).^2);      %sum of squared errors
    SST = sum((y - mean(y)).^2);    %total sum of squares
    R2 = 1 - SSE/SST;
end

function [a, b, R2] = exponential_fit(x, y)
    %linearize y-values
    y_ln = log(y);
    
    %design matrix A for linearized model
    A = [ones(length(x), 1), x];
    
    %solve for parameters of the linearized model using normal equations
    params = (A' * A) \ (A' * y_ln);
    
    %extract the parameters b and c from the solution
    c = params(1);
    b = params(2);
    
    %convert c to a
    a = exp(c);
    
    %calculate using the exponential model
    y_pred = a * exp(b * x);
    
    %compute R^2
    SSE = sum((y - y_pred).^2);      %sum of squared errors
    SST = sum((y - mean(y)).^2);    %total sum of squares
    R2 = 1 - SSE/SST;
end

function [a, b, R2] = saturation_fit(x, y)
    %reciprocal transformation
    Y_prime = 1 ./ y;
    X_prime = 1 ./ x;
    
    %design matrix A for the transformed linear model
    A = [X_prime, ones(length(x), 1)];
    
    %solve for parameters using the normal equations
    params = (A' * A) \ (A' * Y_prime);
    
    %extract parameters
    c = params(1);
    m = params(2);
    
    %convert parameters m and c to a and b
    a = 1 / m;
    b = c * a;
    
    %calculate predicted values using the saturation model
    y_pred = (a * x) ./ (b + x);
    
    %compute R^2
    SSE = sum((y - y_pred).^2);      %sum of squared errors
    SST = sum((y - mean(y)).^2);    %total sum of squares
    R2 = 1 - SSE/SST;
end
