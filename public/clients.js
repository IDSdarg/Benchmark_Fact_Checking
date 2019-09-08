$.ajaxSetup({
    timeout: 30000000, 
    retryAfter:70000000
});



$(function(){				
        $('#kb').on('change',function(e){
            
            e.preventDefault();
            console.log('success');
			var data = {};
			data.kb = $('#kb').val();
			$.ajax({
				type: 'POST',
				data: data,
                url: '/kb_process',						
                success: function(data) {
                    console.log('success');
                    $('<option disabled selected value>').text("-- select a predicate --").appendTo('#predicate');
                    $.each(data, function (x) {
                        $('<option>').val(data[x]).text(data[x]).appendTo('#predicate');
                    })
           
                }
            });

        });	

        $('#predicate').on('change', function(e){
            e.preventDefault();
            console.log('Filter triples having this relation');
            // $('#c_predicate').empty();
            var data = {};
            data.predicate = $('#predicate').val();
            data.kb = $('#kb').val();
            $.ajax({
                type: 'POST',
                data: data,
                url: '/predicate_process',                     
                beforeSend: function() {
                    $("#loading_predicate").show();
                    $('#output_predicate').attr("style","display:none");
                },                 
                success: function(data) {
                    $("#loading_predicate").hide();
                    console.log('success');
                    $('#output_predicate').attr("style","display:block");
                    $("#info_predicate").html(data["info"]);
                   
                }
            });
        });
        
        $('#train_level').on('change', function(e){
            e.preventDefault();
            console.log('... Selecting level ...');
            var lv = $('#train_level').val();
            if (lv == "easy") {
                $('#train_consist').prop('value', 1.0);
                $('#train_homo').prop('value', 1.0);
                $('#train_consist').prop('disabled', true);
                // $('#c_predicate').attr("disabled", true);

            } else if (lv == "hard") {
                $('#train_consist').prop('value', 0.5);
                $('#train_homo').prop('value', 0.5);
                $('#train_consist').prop('disabled', false);
                // $('#c_predicate').attr("disabled", false);
            } else if (lv == "medium") {
                $('#train_consist').prop('value', 0.8);
                $('#train_homo').prop('value', 0.8);
                $('#train_consist').prop('disabled', false);
                // $('#c_predicate').attr("disabled", false);
            } 
        });

        $('#test_level').on('change', function(e){
            e.preventDefault();
            console.log('... Selecting level ...');
            var lv = $('#test_level').val();
            if (lv == "easy") {
                $('#test_consist').prop('value', 1.0);
                $('#test_homo').prop('value', 1.0);
                $('#test_consist').prop('disabled', true);
                // $('#c_predicate').attr("disabled", true);

            } else if (lv == "hard") {
                $('#test_consist').prop('value', 0.5);
                $('#test_homo').prop('value', 0.5);
                $('#test_consist').prop('disabled', false);
                // $('#c_predicate').attr("disabled", false);
            } else if (lv == "medium") {
                $('#test_consist').prop('value', 0.8);
                $('#test_homo').prop('value', 0.8);
                $('#test_consist').prop('disabled', false);
                // $('#c_predicate').attr("disabled", false);
            } 
        });

        $('#algorithm').on('change', function(e){
            e.preventDefault();
            console.log('Load setting for algorithm...');
            $('#kl_1').attr("style","display:none");
            $('#kl_2').attr("style","display:none");
            $('#kl_3').attr("style","display:none");
            $('#kg_1').attr("style","display:none");
            $('#sfe_1').attr("style","display:none");
            $('#pra_1').attr("style","display:none");
            $('#pra_2').attr("style","display:none");
            $('#trans_1').attr("style","display:none");
            $('#trans_2').attr("style","display:none");
            var algo = $('#algorithm').val();

            if (algo == "kl") {
                $('#kl_1').attr("style","display:block");
                $('#kl_2').attr("style","display:block");
                $('#kl_3').attr("style","display:block");           
            } else if (algo == "kgminer") {
                $('#kg_1').attr("style","display:block");  
                $('#kl_2').attr("style","display:block"); 
            } else if (algo == "pra") {
                $('#pra_1').attr("style","display:block");
                $('#pra_2').attr("style","display:block");
            } else if (algo == "sfe") {
                $('#sfe_1').attr("style","display:block");
            }

        });

        $('#algorithm_2').on('change', function(e){
            e.preventDefault();
            console.log('Load setting for algorithm...');
            $('#kl_12').attr("style","display:none");
            $('#kl_22').attr("style","display:none");
            $('#kl_32').attr("style","display:none");
            $('#kg_12').attr("style","display:none");
            $('#sfe_12').attr("style","display:none");
            $('#pra_12').attr("style","display:none");
            $('#pra_22').attr("style","display:none");
            $('#trans_12').attr("style","display:none");
            $('#trans_22').attr("style","display:none");
            var algo = $('#algorithm_2').val();

            if (algo == "kl") {
                $('#kl_12').attr("style","display:block");
                $('#kl_22').attr("style","display:block");
                $('#kl_32').attr("style","display:block");           
            } else if (algo == "kgminer") {
                $('#kg_12').attr("style","display:block");  
                $('#kl_22').attr("style","display:block"); 
            } else if (algo == "pra") {
                $('#pra_12').attr("style","display:block");
                $('#pra_22').attr("style","display:block");
            } else if (algo == "sfe") {
                $('#sfe_12').attr("style","display:block");
            }

        });

        $('#train_generate').unbind().click(function(e){
            e.preventDefault();        
            var data = {};
            $('#output_train_scenario').attr("style","display:none");
            data.kb = $('#kb').val();
            data.mode = "train";
            data.predicate = $('#predicate').val();
            // data.relate_predicate = $('#c_predicate').val();
            // var optionValues = [];
            // $('#c_predicate option').each(function() {
            //     optionValues.push($(this).val());
            // });       
            // data.relate_predicate = optionValues;   
            // data.level = $('#level').val();     
            data.test_size_scenario = $('#test_size_scenario').val(); 
            data.test_consist = $('#test_consist').val(); 
            data.test_homo = $('#test_homo').val(); 
            data.test_pos_neg  = $('#test_pos_neg').val(); 
            data.test_popular =  $('#test_popular').val(); 
            data.train_size_scenario = $('#train_size_scenario').val();
            data.train_consist = $('#train_consist').val(); 
            data.train_homo = $('#train_homo').val(); 
            data.train_pos_neg  = $('#train_pos_neg').val(); 
            data.train_popular =  $('#train_popular').val();
            $.ajax({
                type: 'POST',
                data: data,
                url: '/generate',
                timeout: 5000000,    
                beforeSend: function() {
                    $("#loading_train").show();
                },                 
                success: function(data) {
                    $("#loading_train").hide();
                    console.log('success');
                    $('#output_train_scenario').attr("style","display:block");
                    $('#train').attr("href", data["output"]);
                    // $("#info_train_generate").html(data["info"]);
                   
                }
            });
        });

        $('#test_generate').unbind().click(function(e){
            e.preventDefault();        
            var data = {};
            $('#output_test_scenario').attr("style","display:none");
            data.kb = $('#kb').val();
            data.mode = "test";
            data.predicate = $('#predicate').val();
            // data.relate_predicate = $('#c_predicate').val();
            // var optionValues = [];
            // $('#c_predicate option').each(function() {
            //     optionValues.push($(this).val());
            // });       
            // data.relate_predicate = optionValues;   
            // data.level = $('#level').val();     
            data.test_size_scenario = $('#test_size_scenario').val(); 
            data.test_consist = $('#test_consist').val(); 
            data.test_homo = $('#test_homo').val(); 
            data.test_pos_neg  = $('#test_pos_neg').val(); 
            data.test_popular =  $('#test_popular').val(); 
            data.train_size_scenario = $('#train_size_scenario').val();
            data.train_consist = $('#train_consist').val(); 
            data.train_homo = $('#train_homo').val(); 
            data.train_pos_neg  = $('#train_pos_neg').val(); 
            data.train_popular =  $('#train_popular').val(); 
            $.ajax({
                type: 'POST',
                data: data,
                url: '/generate',    
                beforeSend: function() {
                    $("#loading_test").show();
                },                 
                success: function(data) {
                    $("#loading_test").hide();
                    console.log('success');
                    $('#output_test_scenario').attr("style","display:block");
                    $('#test').attr("href", data["output"]);
                    // $("#info_test_generate").html(data["info"]);
                   
                }
            });
        }); 	

        $('#fact_check').unbind().click(function(e){
            e.preventDefault();
            
            var data = {};
            
            data.kb = $('#kb').val();
            data.algorithm = $('#algorithm').val();
            data.predicate = $('#predicate').val();
            data.level = $('#level').val();  
            data.closure = $('#closure').val();
            data.is_directed = $('#is_directed').val();
            data.max_depth = parseInt($('#max_depth').val());
            data.subgraph_depth = parseInt($('#subgraph_depth').val());
            data.walksource = $('#w_s').val();
            data.walkpath = $('#w_p').val();
            data.embedsize = $('#e_s').val();
            data.learnrate = $('#l_r').val();
            data.weighttype = $('#weight').val();

            // data.n_procs = $('#n_procs').val();    
            data.test_size_scenario = $('#test_size_scenario').val(); 
            data.test_consist = $('#test_consist').val(); 
            data.test_homo = $('#test_homo').val(); 
            data.test_pos_neg  = $('#test_pos_neg').val(); 
            data.test_popular =  $('#test_popular').val(); 
            data.train_size_scenario = $('#train_size_scenario').val();
            data.train_consist = $('#train_consist').val(); 
            data.train_homo = $('#train_homo').val(); 
            data.train_pos_neg  = $('#train_pos_neg').val(); 
            data.train_popular =  $('#train_popular').val(); 
            $.ajax({
                type: 'POST',
                data: data,
                url: '/fact_check',    
                beforeSend: function() {
                    $("#loading").show();
                    $("#poster").attr("src", "");
                    $('#output').attr("style","display:none");

                },                   
                success: function(data) {
                    console.log('success');
                    $("#loading").hide();
                    $('#output').attr("style","display:block");
                    $('#spec').attr("href", data["spec"]);
                    $('#score').attr("href", data["output"]);
                    $("#auroc").html("AUROC = " + data["auroc"]);
                    $("#poster").attr("src", "roc_curve.svg?"+ new Date().getTime());
                    // $('#poster').attr('src', 'color.php?' + frm.serialize() + '&_' + new Date().getTime() );
                }
            });
        }); 

        $('#fact_check_2').unbind().click(function(e){
            e.preventDefault();
            
            var data = {};
            
            data.kb = $('#kb').val();
            data.algorithm = $('#algorithm_2').val();
            data.predicate = $('#predicate').val();
            data.level = $('#level').val();  
            data.closure = $('#closure_2').val();
            data.is_directed = $('#is_directed_2').val();
            data.max_depth = parseInt($('#max_depth_2').val());
            data.subgraph_depth = parseInt($('#subgraph_depth_2').val());
            data.walksource = $('#w_s_2').val();
            data.walkpath = $('#w_p_2').val();
            data.embedsize = $('#e_s_2').val();
            data.learnrate = $('#l_r_2').val();
            data.weighttype = $('#weight_2').val();

            // data.n_procs = $('#n_procs').val();    
            data.test_size_scenario = $('#test_size_scenario').val(); 
            data.test_consist = $('#test_consist').val(); 
            data.test_homo = $('#test_homo').val(); 
            data.test_pos_neg  = $('#test_pos_neg').val(); 
            data.test_popular =  $('#test_popular').val(); 
            data.train_size_scenario = $('#train_size_scenario').val();
            data.train_consist = $('#train_consist').val(); 
            data.train_homo = $('#train_homo').val(); 
            data.train_pos_neg  = $('#train_pos_neg').val(); 
            data.train_popular =  $('#train_popular').val(); 
            $.ajax({
                type: 'POST',
                data: data,
                url: '/fact_check',    
                beforeSend: function() {
                    $("#loading_2").show();
                    $("#poster_2").attr("src", "");
                    $('#output_2').attr("style","display:none");
                },                   
                success: function(data) {
                    console.log('success');
                    $("#loading_2").hide();
                    $('#output_2').attr("style","display:block");
                    $('#spec_2').attr("href", data["spec"]);
                    $('#score_2').attr("href", data["output"]);
                    $("#auroc_2").html("AUROC = " + data["auroc"]);
                    $("#poster_2").attr("src", "roc_curve.svg?"+ new Date().getTime());
                    
                }
            });
        }); 		
     
    });


$(document).ready(function(){
 
  $('a.spec').click(function(){
    window.open(this.href);
    return false;
  });

  $('a.train').click(function(){
    window.open(this.href);
    return false;
  });
 
});