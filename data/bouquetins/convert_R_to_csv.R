load("bouquetins.RData")

write.csv(scenario, file="scenario.csv")

write.csv(SORTIE[, "PopSize1", "2029", ], file="PopSize1_2029.csv")
write.csv(SORTIE[, "nI", "2029", ], file="nI_2029.csv")

