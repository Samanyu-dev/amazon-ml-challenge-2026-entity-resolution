# Amazon ML Challenge 2026 — dataset statistics
Computed from the raw organiser files (train/test, sources 1–3) and our pipeline artifacts. 27 Sep 2026.

## 1. Files and rows
|  | US | India | France | total | columns |
|---|---|---|---|---|---|
| train S1 | 1,323,633 | 883,188 | 0 | 2,206,821 | entity_id, business_name, business_address, country |
| train S2 | 3,016,817 | 2,017,799 | 0 | 5,034,616 | entity_id, business_name, business_address, country |
| train S3 | 3,170,056 | 2,115,547 | 0 | 5,285,603 | entity_id, business_name, business_address, country |
| test S1 | 663,106 | 809,986 | 259,452 | 1,732,544 | entity_id, business_name, business_address, country |
| test S2 | 1,871,330 | 2,312,565 | 703,378 | 4,887,273 | entity_id, business_name, business_address, country |
| test S3 | 1,945,701 | 2,405,000 | 731,615 | 5,082,316 | entity_id, business_name, business_address, country |

ID format: `S1-<9 digits>`, `S2-<9 digits>`, `S3-<9 digits>`; IDs unique within each file: train S1 True, train S2 True, train S3 True, test S1 True, test S2 True, test S3 True

## 2. Density (S2+S3 records per S1 entity)
|  | S1 | S2+S3 | per S1 | S2 per S1 | S3 per S1 |
|---|---|---|---|---|---|
| train US | 1323633.000 | 6186873.000 | 4.674 | 2.279 | 2.395 |
| train India | 883188.000 | 4133346.000 | 4.680 | 2.285 | 2.395 |
| test US | 663106.000 | 3817031.000 | 5.756 | 2.822 | 2.934 |
| test India | 809986.000 | 4717565.000 | 5.824 | 2.855 | 2.969 |
| test France | 259452.000 | 1434993.000 | 5.531 | 2.711 | 2.820 |

## 3. Ground truth (train)
- True links: **7,638,365** (S2 3,693,619, S3 3,944,746); mean per S1 **3.461**, median 3, max 11
- Singletons (S1 with no copy): **0.0558** (123,247); max S2 copies per S1 5, max S3 copies 6
- Decoys (S2/S3 records with no owner): **0.2599** of records (2,681,854); S2 0.2664, S3 0.2537
- Every record belongs to at most one S1; all links stay within one country.

Copies per S1 entity (share of entities):
|  | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| US | 0.0558 | 0.0542 | 0.1702 | 0.2409 | 0.2194 | 0.1455 | 0.0743 | 0.0289 | 0.0085 | 0.0019 | 0.0003 | 0.0000 |
| India | 0.0559 | 0.0537 | 0.1698 | 0.2400 | 0.2193 | 0.1464 | 0.0753 | 0.0291 | 0.0085 | 0.0019 | 0.0002 | 0.0000 |

S2 copies × S3 copies per entity (share):
|  | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| 0 | 0.0558 | 0.0274 | 0.0258 | 0.0143 | 0.0056 | 0.0014 | 0.0001 |
| 1 | 0.0266 | 0.1222 | 0.1138 | 0.0636 | 0.0247 | 0.0061 | 0.0005 |
| 2 | 0.0220 | 0.1011 | 0.0943 | 0.0526 | 0.0205 | 0.0050 | 0.0004 |
| 3 | 0.0113 | 0.0518 | 0.0479 | 0.0269 | 0.0106 | 0.0025 | 0.0002 |
| 4 | 0.0040 | 0.0184 | 0.0174 | 0.0094 | 0.0037 | 0.0009 | 0.0001 |
| 5 | 0.0008 | 0.0037 | 0.0035 | 0.0019 | 0.0008 | 0.0002 | 0.0000 |

Decoy share by country: US 0.2600, India 0.2597

## 4. Text properties (share of records unless "mean")
|  | name empty | address empty | name chars (mean) | name words (mean) | address chars (mean) | address words (mean) | name ASCII only | name accented Latin | name Indic script | address Indic script | name UPPERCASE | name lowercase | address has digit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| train S1 US | 0.000 | 0.000 | 22.465 | 3.419 | 34.962 | 5.898 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| train S1 India | 0.000 | 0.000 | 26.386 | 3.741 | 77.700 | 11.229 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.913 |
| train S2 US | 0.000 | 0.037 | 23.554 | 3.354 | 31.531 | 5.422 | 0.933 | 0.067 | 0.000 | 0.000 | 0.215 | 0.067 | 0.902 |
| train S2 India | 0.000 | 0.029 | 27.420 | 3.710 | 68.196 | 10.071 | 0.721 | 0.044 | 0.235 | 0.237 | 0.149 | 0.044 | 0.913 |
| train S3 US | 0.000 | 0.035 | 23.977 | 3.416 | 38.445 | 5.879 | 0.932 | 0.068 | 0.000 | 0.000 | 0.032 | 0.069 | 0.913 |
| train S3 India | 0.000 | 0.031 | 27.038 | 3.689 | 59.106 | 9.109 | 0.815 | 0.053 | 0.132 | 0.225 | 0.026 | 0.052 | 0.901 |
| test S1 US | 0.000 | 0.000 | 22.463 | 3.417 | 34.966 | 5.899 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| test S1 India | 0.000 | 0.000 | 26.375 | 3.740 | 77.714 | 11.229 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.913 |
| test S1 France | 0.000 | 0.000 | 19.423 | 3.112 | 50.071 | 7.218 | 0.843 | 0.157 | 0.000 | 0.000 | 0.000 | 0.000 | 0.996 |
| test S2 US | 0.000 | 0.029 | 24.285 | 3.478 | 31.843 | 5.479 | 0.938 | 0.062 | 0.000 | 0.000 | 0.204 | 0.060 | 0.921 |
| test S2 India | 0.000 | 0.023 | 28.251 | 3.821 | 68.745 | 10.165 | 0.724 | 0.040 | 0.236 | 0.238 | 0.141 | 0.038 | 0.930 |
| test S2 France | 0.000 | 0.031 | 21.103 | 3.101 | 39.548 | 6.187 | 0.755 | 0.245 | 0.000 | 0.000 | 0.208 | 0.065 | 0.931 |
| test S3 US | 0.000 | 0.028 | 24.619 | 3.527 | 38.838 | 5.936 | 0.936 | 0.064 | 0.000 | 0.000 | 0.032 | 0.063 | 0.929 |
| test S3 India | 0.000 | 0.025 | 27.867 | 3.801 | 59.420 | 9.177 | 0.818 | 0.049 | 0.133 | 0.229 | 0.025 | 0.046 | 0.919 |
| test S3 France | 0.000 | 0.029 | 21.140 | 3.109 | 39.969 | 6.231 | 0.761 | 0.239 | 0.000 | 0.000 | 0.056 | 0.066 | 0.934 |

Indic scripts in India target names (count):
|  | Devanagari | Bengali | Gurmukhi | Gujarati | Odia | Tamil | Telugu | Kannada | Malayalam |
|---|---|---|---|---|---|---|---|---|---|
| train | 427,427 | 48,867 | 10,794 | 48,949 | 11,810 | 53,571 | 62,356 | 59,206 | 29,889 |
| test | 490,171 | 56,418 | 12,537 | 55,891 | 13,732 | 61,761 | 71,869 | 69,738 | 35,128 |

## 5. Legal form at end of S1 name (share of S1)
- **train US**: (none) 0.451, LLC 0.285, INC 0.18, CORP 0.02, PC 0.017, PLLC 0.016, LP 0.012, CO 0.006
- **train India**: PRIVATE LIMITED 0.489, (none) 0.158, PVT LTD 0.138, LIMITED 0.102, LLP 0.044, LTD 0.028, CO 0.019, COMPANY 0.008
- **test US**: (none) 0.452, LLC 0.285, INC 0.18, CORP 0.02, PC 0.016, PLLC 0.015, LP 0.013, CO 0.006
- **test India**: PRIVATE LIMITED 0.489, (none) 0.158, PVT LTD 0.137, LIMITED 0.101, LLP 0.044, LTD 0.028, CO 0.019, CORPORATION 0.008
- **test France**: (none) 0.311, SARL 0.283, SAS 0.201, EURL 0.065, SA 0.049, SASU 0.041, SCI 0.032, EI 0.016

## 6. Namesakes and shared addresses in S1
|  | S1 | in namesake group (same core name) | mean group size (if >1) | max group | distinct core names | address shared with another S1 | S1 address empty |
|---|---|---|---|---|---|---|---|
| train US | 1323633.000 | 0.431 | 44.385 | 523.000 | 881432.000 | 0.056 | 0.000 |
| train India | 883188.000 | 0.533 | 35.428 | 194.000 | 482323.000 | 0.049 | 0.000 |
| test US | 663106.000 | 0.355 | 27.850 | 253.000 | 482592.000 | 0.040 | 0.000 |
| test India | 809986.000 | 0.526 | 33.049 | 183.000 | 447663.000 | 0.048 | 0.000 |
| test France | 259452.000 | 0.344 | 12.933 | 205.000 | 192305.000 | 0.111 | 0.000 |

## 7. Generator operations on S2/S3 records
Share of records carrying each operation; "true-copy rate" = share of train records with that operation that belong to an S1 (the rest are decoys).
|  | train true-copy rate | train US | train India | test US | test India | test France |
|---|---|---|---|---|---|---|
| blank address | 0.9772 | 0.0359 | 0.0297 | 0.0289 | 0.0237 | 0.0300 |
| made-up syllable name | 0.9789 | 0.0100 | 0.0107 | 0.0082 | 0.0086 | 0.0118 |
| acronym name (2-4 caps) | 0.9962 | 0.0009 | 0.0024 | 0.0007 | 0.0019 | 0.0166 |
| domain name (.com) | 0.9622 | 0.0434 | 0.0356 | 0.0358 | 0.0288 | 0.0346 |
| alias (dba/aka/formerly/t/a) | 0.9997 | 0.0135 | 0.0085 | 0.0109 | 0.0068 | 0.0072 |
| Indic-script name | 0.7322 | 0.0000 | 0.1821 | 0.0000 | 0.1838 | 0.0000 |
| filler word appended | 0.6514 | 0.0905 | 0.0634 | 0.1004 | 0.0612 | 0.0709 |
| honorific prefix | 0.7378 | 0.0000 | 0.0795 | 0.0000 | 0.0796 | 0.0001 |
| legal form moved to front | 0.6764 | 0.0162 | 0.0292 | 0.0170 | 0.0310 | 0.0343 |
| bracketed word | 0.7634 | 0.0655 | 0.0983 | 0.0636 | 0.0976 | 0.1012 |
| id/phone suffix | 0.8056 | 0.0099 | 0.0080 | 0.0094 | 0.0077 | 0.0000 |
| OCR swap (lnc, 5ervice, c0m) | 0.8054 | 0.0174 | 0.0180 | 0.0167 | 0.0167 | 0.0019 |
| address ## | 0.6951 | 0.0255 | 0.0290 | 0.0260 | 0.0306 | 0.0000 |
| address null/N/A | 0.7252 | 0.0376 | 0.0284 | 0.0381 | 0.0288 | 0.0000 |
| address hyphenated number | 0.7180 | 0.0003 | 0.0517 | 0.0003 | 0.0531 | 0.0001 |
| address leading-zero number | 0.7324 | 0.0492 | 0.0597 | 0.0502 | 0.0590 | 0.0329 |
| address CITY/CDP/County | 0.7260 | 0.0689 | 0.0231 | 0.0702 | 0.0234 | 0.0000 |
| name UPPERCASE | 0.7931 | 0.1215 | 0.0862 | 0.1164 | 0.0818 | 0.1306 |

## 8. Our pipeline on this data
- Candidate pairs (blocking): 117,097,579 (67.6 per S1); v8 adds 1,333,971 from address-number and acronym blocking. Recall of true links in candidates: 98.48% (train).
- Stage-2 uncertain band: q in [0.005, 0.999].
- Validation (US/India, folds 90–99, 220,938 entities), leak-free: stage 2 0.988409 → + meta 0.988688; Indic rescue +0.000431; acronym rule +0.00002 (US/India).
- Leaderboard: v7d 0.984931; v7d + acronym rule **0.985608**.
v8 submission, per country:
|  | S1 | links | links per S1 | empty (no link) |
|---|---|---|---|---|
| US | 663106.0000 | 2250528.0000 | 3.3939 | 0.0578 |
| India | 809986.0000 | 2722133.0000 | 3.3607 | 0.0588 |
| France | 259452.0000 | 861360.0000 | 3.3199 | 0.0584 |

Validation loss decomposition (leak-free, 2,561 entity-points lost): partial misses 61%; entities with copies but nothing predicted 21%; entities with a false link 15%; singletons wrongly linked 3%. Blank-address namesake copies ≈ 0.0051 of the 0.0116 loss.