import pathlib
from typing import Dict, List, Literal

from deepsysid.models.base import DynamicIdentificationModelConfig
from deepsysid.pipeline.configuration import ExperimentConfiguration
from deepsysid.pipeline.evaluation import evaluate_4dof_ship_trajectory, evaluate_model
from deepsysid.pipeline.testing import test_model as run_model
from deepsysid.pipeline.training import train_model


def get_data(idx: int) -> str:
    data = [
        """time,n,deltal,deltar,Vw,alpha_x,alpha_y,u,v,p,r,phi
0,366.361745840294,-0.000667792634540374,0.000505048081770158,2.3058404686817005,0.730398366799378,0.6830213948162981,1.9059304828813899,0.0684893236802277,0.0,0.0,0.0
1,376.009382965212,0.0269512428759281,-0.0275011392958471,2.3710182821464,0.727234974709556,0.6863885864138412,2.03884441675177,0.18236857829187098,0.000587684355071629,-0.0036830559064337702,0.000936654084037869
2,382.34862259643,0.0399361912676506,-0.0386306646481679,2.38084458528952,0.723730265315054,0.6900829682487479,1.75950097621673,-0.0907581959044003,0.0006082768166230059,0.00044169002253877196,0.000435782455662448
3,389.835857065013,0.046755325565030995,-0.0476111408237287,2.01843484456215,0.7251342690183051,0.688607502061434,1.8462101011187,-0.0129042181621594,0.0032779386787100103,-0.0023536959270990102,0.00386953909692567
4,397.31677097260905,0.0465441406483103,-0.0470695790336945,2.33779539530011,0.7334518382840741,0.6797414220993989,2.01280015548737,0.141466073569668,-0.00285088013158514,-0.00197233211138525,0.00319969463721392
5,411.037526153852,0.0475113622931288,-0.0478837999799709,2.2579219525721004,0.725006329292575,0.688742203212281,1.9047863152502098,0.032017955996823,-0.0004908282313401509,-3.38223622529014e-05,0.00156485107290116
6,419.834839809093,0.0476024129876059,-0.0475745525927269,2.09954646541522,0.7463645963071289,0.665537293755426,1.9694847193221,0.0819232145703766,-0.0017846855220480602,-0.0016273348867198699,0.000865956187296124
7,431.362052087903,0.0492558766870364,-0.0472582689380661,1.9292578042825401,0.7512140476399749,0.6600586751406011,1.8406128022701,-0.0601186496833452,0.0018434444745084398,-0.00191870528947524,-0.000214538485454607
8,440.001269939275,0.0505244210016282,-0.0491827914164941,2.2348180624252003,0.750048207665694,0.6613831613954799,1.80209581087529,-0.10596250199770699,6.52428567097881e-05,-0.000134102825985837,0.00185243119797418
9,448.01114330711096,0.0475339737771629,-0.0471199403213803,2.4150973185285096,0.794690353250433,0.607015026544403,1.9421733156566299,0.0102059143100228,0.00199819766511667,-0.00291369272389946,0.00175095253104463
10,453.779241630088,0.0510934717190151,-0.0473708812140114,2.52585588320771,0.785814733641072,0.6184619668117121,1.9827173295557499,0.0396450907678682,-0.0006267600417853259,-0.000243546019181656,0.00303212683318588
11,469.899137029269,0.0479787596364806,-0.049357300283615,2.42101943661268,0.777615519733192,0.628740092144662,1.9010992498300099,-0.0605564510217396,-0.00291966822607557,-0.0010748929753576699,0.0007322554691207019
12,481.455921169419,0.0482615570713931,-0.0482176587124583,2.26130023352433,0.7440363943080659,0.668139090268674,1.8973496065086002,-0.0909402651126372,0.00254079738926358,-0.00426160596718087,0.000480278493763642
13,496.868072923762,0.0495610667574923,-0.04789348086903599,2.44388119339854,0.745244170547202,0.666791666313709,1.9698747587019598,-0.0287622691600405,0.000284020704586125,6.5191979181466795e-06,0.00285572237558318
14,492.06966843026703,0.0477390520516743,-0.0480920960439072,2.36462928249883,0.738092911851911,0.6746990836469,2.00638317109559,-0.00912760523045755,-0.0013898607423967302,0.0006906044450530179,0.0012929558203169699
15,510.28828476126404,0.0480895981835682,-0.0480341395626073,2.3924964658650096,0.7055382483431201,0.708671842339543,2.19829687061914,0.14681771627624898,-0.000733501221043244,-0.00323433427988446,0.000293903130811568
16,519.72845676638,0.0494187789415763,-0.0500312747250407,2.3670946589473902,0.7427016782925621,0.66962244366614,2.19671240042976,0.125964175041588,0.0014544605924601,-0.0013991678566734,0.00047942210056506297
17,528.440210831323,0.0486114894740257,-0.0449484797858022,2.2239290630314104,0.7426607316361321,0.6696678562434399,1.8855781702953702,-0.213260381534752,-0.00019172120483289698,-0.0018320427633703698,0.0015216453441886401
18,540.736157472181,0.0481167108185372,-0.04731957304601599,2.3536633049014704,0.7726003403600129,0.634892679179397,1.95621171766204,-0.16660011376941,0.00152142558668708,-0.00168753082629437,0.0027489586480538
19,554.7193552566259,0.048357490543737995,-0.048641062031917,2.09795573215677,0.7824316658348991,0.622736451718401,2.18252547728919,0.0262853145885283,0.000342950387680747,-0.00245570754581566,0.00230577116068575
20,557.299080150986,0.0480849372093471,-0.0496190122688432,2.04730215212223,0.786247230461866,0.617912042762596,2.15990846414575,-0.0221439876298504,-0.00224386046851402,-0.000991772358433273,0.00171106129909671
""",
        """time,n,deltal,deltar,Vw,alpha_x,alpha_y,u,v,p,r,phi
21,564.3107944310141,0.0489098535851955,-0.0499450500761649,2.2325448549184905,0.785451730767589,0.61892291816849,2.1235568505213003,-0.0939716989566955,0.0018208408610357102,-0.00346762084207453,0.00128992548435836
22,581.435747138284,0.0471868286210771,-0.0494624215284578,2.0825598430425702,0.791563363203636,0.611087098565948,2.1382475138205304,-0.10785727694437301,0.00109157037786951,-0.00179984710739245,0.0033771152219848103
23,585.863942338357,0.0484304808249975,-0.0502449934662426,2.11947136788917,0.751423159658476,0.659820608293552,2.22430816353167,-0.05088802470871201,-0.00113574960044098,-1.52621308365223e-05,0.0025745508462775603
24,592.357095954986,0.0495590862538259,-0.0487453486559155,2.37692514231587,0.764606792330707,0.6444970543933821,2.1980948320543603,-0.11480458998541498,-0.00102585252829535,-0.00231787088216553,0.00253343507346804
25,610.813325995469,0.0492209392028801,-0.0476745619043104,2.25598159879732,0.755596671196924,0.655037151981571,2.3783000426212597,0.0268326310539142,-0.0041953405872452295,-0.000686208485274425,-0.00112859374965346
26,623.2146864672831,0.0502642654992404,-0.0500649464612156,2.04226966449034,0.7451973180394359,0.6668440276307739,2.5370801871700697,0.14996419191978302,0.00120634568413891,-0.00041786510931677797,-0.00148749123887759
27,629.973509865586,0.0497467382377424,-0.049018132214289,2.11473024342906,0.75398724418404,0.656889058827864,2.51593126506447,0.0841134527967458,0.00293564881583197,-0.0018456805277939198,0.000153474064364278
28,638.338062289309,0.0499527618390601,-0.0473071149134004,2.0342217952031003,0.7503942884661059,0.660990477871994,2.5478041702581002,0.0798424647292343,0.0034077817189130397,-0.000878104049765945,0.00308508236733473
29,651.503533264615,0.0472760229296414,-0.0496264779540709,2.0944133601411,0.7683225105231899,0.640062903020745,2.52443703545437,0.0149362260770117,-0.0008772381169774151,-0.000551750039108445,0.00422955349047749
30,662.723510801128,0.0466345507842106,-0.0487777129525742,2.0452868987414,0.750064522094947,0.661364659393348,2.41680023669542,-0.13152557316196303,-0.0022760829620211698,-0.00023496838893635201,0.00395929276189966
31,666.683813903983,0.0497653765621995,-0.0497444401826025,1.9150571790881599,0.789517091670385,0.613728573524434,2.4244273914902004,-0.177506312631962,-0.0028780074190809897,-0.00213026218273713,0.000907581868895502
32,676.168193087789,0.0481945292373757,-0.0489530927387408,1.8550384668683402,0.748834756902453,0.6627567478757529,2.79806638164975,0.156838936740067,0.00046815402246421905,-0.00156184612541957,-0.00043008799880084304
33,690.726603767494,0.0480793187551375,-0.0477035916653254,2.0346057135146003,0.7319936015457299,0.681311505330793,2.7232218268215096,0.0383181923562937,0.004163650106505329,-0.000962944604130817,0.00160443430940216
34,707.10274851918,0.0490902487377042,-0.0472625358026622,2.04517903743056,0.74804422972353,0.6636488758201371,2.64295285251989,-0.0931502037602669,-9.775902980469571e-05,-0.00206282346514767,0.00473063069692614
35,705.049122249327,0.0480516620855833,-0.0492243538507275,2.20436172427571,0.745755867934535,0.66621932232652,2.82143515720516,0.0375958407431133,-0.00310685982455854,-0.000338851563455044,0.00136415417269076
36,725.1835525815429,0.0467957801868756,-0.0495387542547914,2.10117461514138,0.700471372700256,0.713680499962987,2.8646802813984,0.0359379243961616,0.00505722701650636,-0.00330727158306405,0.003060070507789
37,732.980040949776,0.049735143607402,-0.0482757555662046,2.2615825374434997,0.711847048342993,0.7023345212684399,3.0574323139993704,0.18112504378107297,0.00350751807703769,-0.00408497806372406,0.008326151257355809
38,747.665442022873,0.0476028196071043,-0.0471675286115593,2.08749005613369,0.674172246870521,0.738574154401271,2.82963243736869,-0.103883525956029,-0.0048246523269643,-0.00344935971915691,0.00787778869154465
39,750.2143287024479,0.0485435356486737,-0.0499673811653817,2.1677500263187097,0.6565633651801711,0.7542708714402859,3.1174152747957997,0.128439177074828,-0.00959943390656522,-0.0018749600339355102,-1.46921291257485e-05
40,756.762577065957,0.0490262581737373,-0.047776932845968,2.13492408098303,0.660176029988559,0.7511109168615151,3.22509510693856,0.18436231005274697,-0.005688196154280361,0.0007191232275966911,-0.00827355445874645
41,772.0408281644251,0.0503336857725858,-0.0482384264424621,2.2005892873369,0.66722786146666,0.744853664072768,3.1417223607876896,0.0455261140087404,0.00612810143164519,5.24662089428631e-05,-0.00846814650158034
""",
        """time,n,deltal,deltar,Vw,alpha_x,alpha_y,u,v,p,r,phi
42,779.2635275622839,0.0468853911485151,-0.0478154102753596,2.21787400884755,0.665219036391537,0.7466482663358399,3.13895994107151,-0.0137046617807481,0.0102645287565933,-0.000152058626934923,0.000521446018352467
43,789.022927546412,0.0471439487035629,-0.0478147664012548,1.89153447114517,0.6888499433089599,0.724903963020787,3.19082914571521,-0.0148121123841945,0.00616051534502947,-5.2382185474143796e-05,0.0103820082923054
44,800.6603765163161,0.0479852125797857,-0.0475803757919274,1.9982662735018601,0.6990188316190289,0.715103260405075,3.3685858706937,0.11070698831124401,-0.0022273507791980003,0.00016967210795063102,0.0126854407043369
45,810.0462749208169,0.0480359090434938,-0.0477878083932674,1.97442822705701,0.7337435968956791,0.679426474325656,3.04559238003484,-0.267255037853298,-0.00784260653952282,0.00044681388008737695,0.0068376325914880605
46,820.3415461469341,0.0473398665248326,-0.0454620405465667,1.9961749139061498,0.727043916054143,0.6865909583796279,3.47601523442486,0.0988634956813197,-0.0071648641535183,-0.00165904982891938,-0.00186940174452088
47,831.7170910232879,0.0487948568904334,-0.0506951847748682,2.05131821362133,0.7403819376013779,0.672186422410948,3.30347696078805,-0.13304202031598,9.570838810689849e-05,-0.000693268629283579,-0.00579431817690246
48,836.915495858015,0.049438901340214,-0.0481586423689419,1.9976323882872098,0.728956791682802,0.684559709491813,3.4493398891425002,-0.0426420781941031,0.00912982654614647,-0.0010266600144954198,-0.000412595464901244
49,840.2677814517401,0.0498405201854952,-0.0479171021778955,1.9476794690689998,0.734704964718532,0.678386773764009,3.4519490315912598,-0.104626579999396,0.0048715483670076,-0.0011385856811959802,0.00797402853038287
50,856.0107635148421,0.0491486982457547,-0.0488037145149412,2.17035741396947,0.7250123861283589,0.6887358274116879,3.7653862360667203,0.151490329951141,-0.000222907318386342,-0.0011978863916079301,0.00903786083987956
51,867.690589946488,0.048756045552412,-0.0473752070940022,1.9581170152161798,0.681406220832082,0.7319054325603409,3.5688828382577205,-0.096819898753626,-0.0055101506184428,-0.000127673873513424,0.0075958910313072
52,881.862577275917,0.0484617782664631,-0.0498023401968519,2.0956322770523,0.680068979845401,0.733148131452324,3.75013029590742,0.0317939697931826,-0.0061414461728741205,0.00203608313882804,0.000387880895960535
53,891.0335330488,0.0472820306501893,-0.0483449122805428,2.16533180027389,0.6754225672041241,0.7374309158906959,3.9376792081457697,0.151401570549767,0.00303636905833323,-0.00123762392878589,-0.000628337344864338
54,903.010085705154,0.049588343897845,-0.0484130678616242,2.12906249471842,0.65885141195615,0.7522730999865591,3.79033507738183,-0.0612431155562325,0.00352233568095115,-0.0014211358150077302,0.00279079223450074
55,913.4312888519501,0.0517470503858245,-0.0501125418049593,2.0785272828729298,0.663384664663422,0.7482785488635891,3.77286168264359,-0.133728483697076,0.000757257561729081,0.0009050637709503071,0.00537081248621796
56,918.762735187833,0.0470824691602568,-0.0485587536464941,2.1140423553957097,0.6742719858304038,0.738483100093918,4.11978318023555,0.14892783563361098,0.000431067675545431,0.000267987615445264,0.00568321817903605
57,932.09587512621,0.0500344918279096,-0.047994268565317,2.2203487818000003,0.7130266289229201,0.7011369527038321,3.97020186272801,-0.0629837066493498,7.88361544243121e-05,-0.00014950744697415601,0.00535722229905478
58,942.6889443040891,0.0505529187900295,-0.0472126676967679,2.06485265361489,0.700409770358031,0.713740956921355,4.087741846300831,-0.00833019913516923,0.000244404627605965,-0.000541015804193715,0.00522212564291635
59,948.472863284931,0.0476632131049097,-0.0480968653361294,2.08671586351264,0.7237762226733341,0.6900347668724529,4.16934931949569,0.00857470871205421,-0.0005542869728544519,-0.000413522703097472,0.00525395447194166
60,964.270018721296,0.0489632410466911,-0.0489243541187817,1.7367185229698,0.7162282044941529,0.697866146970235,4.188045906921779,-0.0360388584879017,8.421782012194941e-05,-0.000873922589953951,0.00554811659663379
61,966.892211478738,0.0477650626918437,-0.0482912296371558,1.9353854458714,0.715659673741693,0.698449161628772,4.411354720173151,0.116142908929917,5.4867469118364504e-05,-0.00225475301015267,0.0073422420355471205
62,980.6345171475509,0.0477078453856838,-0.046800667435083,1.95440414881116,0.723079128462368,0.690765209012514,4.469737071132231,0.10033878805219,-0.0023162647254260397,-0.00151016889420793,0.00513054948608715
""",
    ]
    return data[idx % len(data)]


def get_state_names() -> List[str]:
    return ['u', 'v', 'p', 'r', 'phi', 'alpha_x', 'alpha_y']


def get_control_names() -> List[str]:
    return ['n', 'deltal', 'deltar', 'Vw']


def get_cpu_device_name() -> str:
    return 'cpu'


def get_window_size() -> int:
    return 3


def get_horizon_size() -> int:
    return 2


def get_time_delta() -> float:
    return 0.5


def get_evaluation_mode() -> Literal['train', 'validation', 'test']:
    return 'test'


def get_thresholds() -> List[float]:
    return [1.0, 0.5, 0.1]


def get_train_fraction() -> float:
    return 0.6


def get_validation_fraction() -> float:
    return 0.3


def prepare_directories(
    base_path: pathlib.Path,
) -> Dict[str, pathlib.Path]:
    models_directory = base_path.joinpath('models')
    models_directory.mkdir(exist_ok=True)

    dataset_directory = base_path.joinpath('data')
    train_directory = dataset_directory.joinpath('processed').joinpath('train')
    validation_directory = dataset_directory.joinpath('processed').joinpath(
        'validation'
    )
    test_directory = dataset_directory.joinpath('processed').joinpath('test')

    dataset_directory.mkdir(exist_ok=True)
    dataset_directory.joinpath('processed').mkdir(exist_ok=True)
    train_directory.mkdir(exist_ok=True)
    validation_directory.mkdir(exist_ok=True)
    test_directory.mkdir(exist_ok=True)

    result_directory = base_path.joinpath('results')
    result_directory.mkdir(exist_ok=True)

    configuration_path = base_path.joinpath('configuration.json')

    return dict(
        models=models_directory,
        data=dataset_directory,
        train=train_directory,
        validation=validation_directory,
        test=test_directory,
        configuration=configuration_path,
        result=result_directory,
    )


def run_pipeline(
    base_path: pathlib.Path,
    model_name: str,
    model_class: str,
    config: DynamicIdentificationModelConfig,
):
    # Define and create temporary file paths and directories.
    paths = prepare_directories(base_path)

    # Setup configuration file.
    configuration_dict = dict(
        train_fraction=get_train_fraction(),
        validation_fraction=get_validation_fraction(),
        time_delta=get_time_delta(),
        window_size=get_window_size(),
        horizon_size=get_horizon_size(),
        control_names=get_control_names(),
        state_names=get_state_names(),
        thresholds=get_thresholds(),
        models={
            model_name: dict(
                model_class=model_class,
                parameters=config.dict(),
            )
        },
    )
    config = ExperimentConfiguration.parse_obj(configuration_dict)

    # Setup dataset directory.
    paths['train'].joinpath('train-0.csv').write_text(data=get_data(0))
    paths['validation'].joinpath('validation-0.csv').write_text(data=get_data(1))
    paths['test'].joinpath('test-0.csv').write_text(data=get_data(2))

    # Run model training.
    train_model(
        model_name=model_name,
        device_name=get_cpu_device_name(),
        configuration=config,
        dataset_directory=str(paths['data']),
        models_directory=str(paths['models']),
    )

    # Run model testing.
    run_model(
        model_name=model_name,
        device_name=get_cpu_device_name(),
        mode=get_evaluation_mode(),
        configuration=config,
        dataset_directory=str(paths['data']),
        result_directory=str(paths['result']),
        models_directory=str(paths['models']),
    )

    # Run model evaluation.
    evaluate_model(
        config=config,
        model_name=model_name,
        mode=get_evaluation_mode(),
        result_directory=str(paths['result']),
        threshold=None,
    )

    evaluate_4dof_ship_trajectory(
        configuration=config,
        result_directory=str(paths['result']),
        model_name=model_name,
        mode=get_evaluation_mode(),
    )
