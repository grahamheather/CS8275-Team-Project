list = dir('R*');
DirLen = length(list);

vowels = ["a_normal","i_normal","u_normal"];

l_vowels = length(vowels);
 D = cell(DirLen,l_vowels);

for k = 1:l_vowels
    for i = 1:DirLen
        f_name = strcat(list(i).folder,'/', list(i).name,'/',vowels(k));

        D{i,k} = get_data(f_name);
     end
end

