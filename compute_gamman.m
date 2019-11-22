M = csvread('/media/data/Datasets/MEOP/MEOP-CTD_2018-04-10/myScripts/STPlonlat.csv', 1, 1);
unique_profiles = unique(M(:, 1) );
gamman = zeros(length(M), 1);

for i = 1:length(unique_profiles)
    profIndex = M(:, 1) == unique_profiles(i); %Boolean index selecting unique profiles
    
    SP = M(profIndex, 2);
    t = M(profIndex, 3);
    p = M(profIndex, 4);
    lon = M(profIndex, 5);
    lat = M(profIndex, 6);
    
    gamman(profIndex) = eos80_legacy_gamma_n(SP, t, p, lon, lat);
    unique_profiles(i)
end

dlmwrite('gamman.csv', [M(:,1), gamman],'precision','%.7f');