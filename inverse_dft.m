function result = inverse_dft(arr)
    %expecting a column vector
    new_arr = zeros(size(arr));
    N = size(arr,1);
    for n = 1:N
        temp_arr = [];
        for k = 1:N
            temp_arr = [temp_arr, arr(k)*exp(1i*2*pi*k*n/N)];
        end
        new_arr(n) = 1/N * sum(temp_arr);
    end
    result = new_arr;