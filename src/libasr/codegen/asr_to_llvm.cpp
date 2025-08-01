#include <iostream>
#include <memory>
#include <unordered_map>
#include <utility>

#include <llvm/ADT/STLExtras.h>
#include <llvm/Analysis/Passes.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/GenericValue.h>
#include <llvm/ExecutionEngine/MCJIT.h>
#include <llvm/IR/Argument.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ManagedStatic.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/ExecutionEngine/ObjectCache.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <llvm/IR/DIBuilder.h>
#if LLVM_VERSION_MAJOR < 18
#   include <llvm/Transforms/Vectorize.h>
#endif

#include <libasr/asr.h>
#include <libasr/containers.h>
#include <libasr/codegen/asr_to_llvm.h>
#include <libasr/pass/pass_manager.h>
#include <libasr/exception.h>
#include <libasr/asr_utils.h>
#include <libasr/codegen/llvm_utils.h>
#include <libasr/codegen/llvm_array_utils.h>
#include <libasr/pass/intrinsic_function_registry.h>

namespace LCompilers {

using ASR::is_a;
using ASR::down_cast;
using ASR::down_cast2;

using ASRUtils::expr_type;
using ASRUtils::symbol_get_past_external;
using ASRUtils::EXPR2VAR;
using ASRUtils::EXPR2FUN;
using ASRUtils::intent_local;
using ASRUtils::intent_return_var;
using ASRUtils::determine_module_dependencies;
using ASRUtils::is_arg_dummy;
using ASRUtils::is_argument_of_type_CPtr;

class ASRToLLVMVisitor : public ASR::BaseVisitor<ASRToLLVMVisitor>
{
private:
  //! To be used by visit_StructInstanceMember.
  std::string current_der_type_name;

    //! Helpful for debugging while testing LLVM code
    void print_util(llvm::Value* v, std::string fmt_chars, std::string endline) {
        // Usage:
        // print_util(tmp, "%d", "\n") // `tmp` is an integer type to match the format specifiers
        std::vector<llvm::Value *> args;
        std::vector<std::string> fmt;
        args.push_back(v);
        fmt.push_back(fmt_chars);
        std::string fmt_str;
        for (size_t i=0; i<fmt.size(); i++) {
            fmt_str += fmt[i];
            if (i < fmt.size()-1) fmt_str += " ";
        }
        fmt_str += endline;
        llvm::Value *fmt_ptr = builder->CreateGlobalStringPtr(fmt_str);
        std::vector<llvm::Value *> printf_args;
        printf_args.push_back(fmt_ptr);
        printf_args.insert(printf_args.end(), args.begin(), args.end());
        printf(context, *module, *builder, printf_args);
    }

    //! Helpful for debugging while testing LLVM code
    void print_util(llvm::Value* v, std::string endline="\n") {
        // Usage:
        // print_util(tmp)
        std::string buf;
        llvm::raw_string_ostream os(buf);
        v->print(os);
        std::cout << os.str() << endline;
    }

    //! Helpful for debugging while testing LLVM code
    void print_util(llvm::Type* v, std::string endline="\n") {
        // Usage:
        // print_util(tmp->getType())
        std::string buf;
        llvm::raw_string_ostream os(buf);
        v->print(os);
        std::cout << os.str() << endline;
    }

public:
    diag::Diagnostics &diag;
    llvm::LLVMContext &context;
    std::unique_ptr<llvm::Module> module;
    std::unique_ptr<llvm::IRBuilder<>> builder;
    std::string infile;
    Allocator &al;

    llvm::Value *tmp;
    llvm::BasicBlock *proc_return;
    std::string mangle_prefix;
    bool prototype_only;
    llvm::StructType *complex_type_4, *complex_type_8;
    llvm::StructType *complex_type_4_ptr, *complex_type_8_ptr;
    llvm::Type* string_descriptor;
    llvm::PointerType *character_type;
    llvm::PointerType *list_type;
    std::vector<std::string> struct_type_stack;

    std::unordered_map<std::uint32_t, std::unordered_map<std::string, llvm::Type*>> arr_arg_type_cache;

    std::map<std::string, std::pair<llvm::Type*, llvm::Type*>> fname2arg_type;

    // Maps for containing information regarding derived types
    std::map<std::string, llvm::StructType*> name2dertype, name2dercontext;
    std::map<std::string, std::string> dertype2parent;
    std::map<std::string, std::map<std::string, int>> name2memidx;

    std::map<uint64_t, llvm::Value*> llvm_symtab; // llvm_symtab_value
    std::map<uint64_t, llvm::Value*> llvm_symtab_deep_copy;
    std::map<uint64_t, llvm::Function*> llvm_symtab_fn;
    std::map<std::string, uint64_t> llvm_symtab_fn_names;
    std::map<uint64_t, llvm::Value*> llvm_symtab_fn_arg;
    std::map<uint64_t, llvm::BasicBlock*> llvm_goto_targets;
    std::set<uint32_t> global_string_allocated;
    const ASR::Function_t *parent_function = nullptr;

    std::vector<llvm::BasicBlock*> loop_head; /* For saving the head of a loop,
        so that we can jump to the head of the loop when we reach a cycle */
    std::vector<std::string> loop_head_names;
    std::vector<llvm::BasicBlock*> loop_or_block_end; /* For saving the end of a block,
        so that we can jump to the end of the block when we reach an exit */
    std::vector<std::string> loop_or_block_end_names;

    int64_t ptr_loads;
    bool lookup_enum_value_for_nonints;
    bool is_assignment_target;
    int64_t global_array_count;

    CompilerOptions &compiler_options;

    // For handling debug information
    std::unique_ptr<llvm::DIBuilder> DBuilder;
    llvm::DICompileUnit *debug_CU;
    llvm::DIScope *debug_current_scope;
    std::map<uint64_t, llvm::DIScope*> llvm_symtab_fn_discope;
    llvm::DIFile *debug_Unit;

    std::map<ASR::symbol_t*, std::map<SymbolTable*, llvm::Value*>> type2vtab;
    std::map<ASR::symbol_t*, std::map<SymbolTable*, std::vector<llvm::Value*>>> class2vtab;
    std::map<ASR::symbol_t*, llvm::Type*> type2vtabtype;
    std::map<ASR::symbol_t*, int> type2vtabid;
    std::map<ASR::symbol_t*, std::map<std::string, int64_t>> vtabtype2procidx;
    // Stores the map of pointer and associated type, map<ptr, i32>, Used by Load or GEP
    std::map<llvm::Value *, llvm::Type *> ptr_type;
    llvm::Type* current_select_type_block_type;
    ASR::ttype_t* current_select_type_block_type_asr;
    std::string current_select_type_block_der_type;
    std::string current_selector_var_name;

    SymbolTable* current_scope;
    std::unique_ptr<LLVMUtils> llvm_utils;
    std::unique_ptr<LLVMList> list_api;
    std::unique_ptr<LLVMTuple> tuple_api;
    std::unique_ptr<LLVMDictInterface> dict_api_lp;
    std::unique_ptr<LLVMDictInterface> dict_api_sc;
    std::unique_ptr<LLVMSetInterface> set_api_lp;
    std::unique_ptr<LLVMSetInterface> set_api_sc;
    std::unique_ptr<LLVMArrUtils::Descriptor> arr_descr;
    std::vector<llvm::Value*> heap_arrays;
    Vec<llvm::Value*> strings_to_be_deallocated;
    struct to_be_allocated_array{ // struct to hold details for the initializing pointer_to_array_type later inside main function.
        ASR::expr_t* expr;
        llvm::Constant* pointer_to_array_type;
        llvm::Type* array_type;
        LCompilers::ASR::ttype_t* var_type;
        size_t n_dims;
    };
    std::vector<to_be_allocated_array> allocatable_array_details;
    struct variable_inital_value { /* Saves information for variables that need to be initialized once. To be initialized in `program`*/
        ASR::Variable_t* v;
        llvm::Value* target_var; // Corresponds to variable `v` in llvm IR.
    };
    std::vector<variable_inital_value> variable_inital_value_vec; /* Saves information for variables that need to be initialized once. To be initialized in `program`*/
    ASRToLLVMVisitor(Allocator &al, llvm::LLVMContext &context, std::string infile,
        CompilerOptions &compiler_options_, diag::Diagnostics &diagnostics) :
    diag{diagnostics},
    context(context),
    builder(std::make_unique<llvm::IRBuilder<>>(context)),
    infile{infile},
    al{al},
    prototype_only(false),
    ptr_loads(2),
    lookup_enum_value_for_nonints(false),
    is_assignment_target(false),
    global_array_count(0),
    compiler_options(compiler_options_),
    current_select_type_block_type(nullptr),
    current_select_type_block_type_asr(nullptr),
    current_scope(nullptr),
    llvm_utils(std::make_unique<LLVMUtils>(context, builder.get(),
        current_der_type_name, name2dertype, name2dercontext, struct_type_stack,
        dertype2parent, name2memidx, compiler_options, arr_arg_type_cache,
        fname2arg_type, ptr_type, llvm_symtab)),
    list_api(std::make_unique<LLVMList>(context, llvm_utils.get(), builder.get())),
    tuple_api(std::make_unique<LLVMTuple>(context, llvm_utils.get(), builder.get())),
    dict_api_lp(std::make_unique<LLVMDictOptimizedLinearProbing>(context, llvm_utils.get(), builder.get())),
    dict_api_sc(std::make_unique<LLVMDictSeparateChaining>(context, llvm_utils.get(), builder.get())),
    set_api_lp(std::make_unique<LLVMSetLinearProbing>(context, llvm_utils.get(), builder.get())),
    set_api_sc(std::make_unique<LLVMSetSeparateChaining>(context, llvm_utils.get(), builder.get())),
    arr_descr(LLVMArrUtils::Descriptor::get_descriptor(context,
              builder.get(), llvm_utils.get(),
              LLVMArrUtils::DESCR_TYPE::_SimpleCMODescriptor, compiler_options_, heap_arrays))
    {
        llvm_utils->tuple_api = tuple_api.get();
        llvm_utils->list_api = list_api.get();
        llvm_utils->dict_api = nullptr;
        llvm_utils->set_api = nullptr;
        llvm_utils->arr_api = arr_descr.get();
        llvm_utils->dict_api_lp = dict_api_lp.get();
        llvm_utils->dict_api_sc = dict_api_sc.get();
        llvm_utils->set_api_lp = set_api_lp.get();
        llvm_utils->set_api_sc = set_api_sc.get();
        strings_to_be_deallocated.reserve(al, 1);
    }

    #define load_non_array_non_character_pointers(expr, type, llvm_value) if( ASR::is_a<ASR::StructInstanceMember_t>(*expr) && \
        !ASRUtils::is_array(type) && \
        LLVM::is_llvm_pointer(*type) && \
        !ASRUtils::is_character(*type) ) { \
        llvm::Type *llvm_type = llvm_utils->get_type_from_ttype_t_util(expr, \
            ASRUtils::extract_type(type), module.get()); \
        llvm_value = llvm_utils->CreateLoad2(llvm_type, llvm_value); \
    } \

    // Inserts a new block `bb` using the current builder
    // and terminates the previous block if it is not already terminated
    void start_new_block(llvm::BasicBlock *bb) {
        llvm::BasicBlock *last_bb = builder->GetInsertBlock();
        llvm::Function *fn = last_bb->getParent();
        llvm::Instruction *block_terminator = last_bb->getTerminator();
        if (block_terminator == nullptr) {
            // The previous block is not terminated --- terminate it by jumping
            // to our new block
            builder->CreateBr(bb);
        }
#if LLVM_VERSION_MAJOR >= 16
        fn->insert(fn->end(), bb);
#else
        fn->getBasicBlockList().push_back(bb);
#endif
        builder->SetInsertPoint(bb);
    }

    template <typename Cond, typename Body>
    void create_loop(char *name, Cond condition, Body loop_body) {

        std::string loop_name;
        if (name) {
            loop_name = std::string(name);
        } else {
            loop_name = "loop";
        }

        std::string loophead_name = loop_name + ".head";
        std::string loopbody_name = loop_name + ".body";
        std::string loopend_name = loop_name + ".end";

        llvm::BasicBlock *loophead = llvm::BasicBlock::Create(context, loophead_name);
        llvm::BasicBlock *loopbody = llvm::BasicBlock::Create(context, loopbody_name);
        llvm::BasicBlock *loopend = llvm::BasicBlock::Create(context, loopend_name);

        loop_head.push_back(loophead);
        loop_head_names.push_back(loophead_name);
        loop_or_block_end.push_back(loopend);
        loop_or_block_end_names.push_back(loopend_name);

        // head
        start_new_block(loophead); {
            llvm::Value* cond = condition();
            builder->CreateCondBr(cond, loopbody, loopend);
        }

        // body
        start_new_block(loopbody); {
            loop_body();
            builder->CreateBr(loophead);
        }

        // end
        loop_head.pop_back();
        loop_head_names.pop_back();
        loop_or_block_end.pop_back();
        loop_or_block_end_names.pop_back();
        start_new_block(loopend);
    }

    void get_type_debug_info(ASR::ttype_t* t, std::string &type_name,
            uint32_t &type_size, uint32_t &type_encoding) {
        type_size = ASRUtils::extract_kind_from_ttype_t(t)*8;
        switch( t->type ) {
            case ASR::ttypeType::Integer: {
                type_name = "integer";
                type_encoding = llvm::dwarf::DW_ATE_signed;
                break;
            }
            case ASR::ttypeType::Logical: {
                type_name = "boolean";
                type_encoding = llvm::dwarf::DW_ATE_boolean;
                break;
            }
            case ASR::ttypeType::Real: {
                if( type_size == 32 ) {
                    type_name = "float";
                } else if( type_size == 64 ) {
                    type_name = "double";
                }
                type_encoding = llvm::dwarf::DW_ATE_float;
                break;
            }
            default : throw LCompilersException("Debug information for the type: `"
                + ASRUtils::type_to_str_python(t) + "` is not yet implemented");
        }
    }

    void debug_get_line_column(const uint32_t &loc_first,
            uint32_t &line, uint32_t &column) {
        LocationManager lm;
        LocationManager::FileLocations fl;
        fl.in_filename = infile;
        lm.files.push_back(fl);
        std::string input;
        if (!read_file(infile, input)) {
            throw CodeGenError("File '" + infile + "' cannot be opened.");
        }
        lm.init_simple(input);
        lm.file_ends.push_back(input.size());
        lm.pos_to_linecol(lm.output_to_input_pos(loc_first, false),
            line, column, fl.in_filename);
    }

    template <typename T>
    void debug_emit_loc(const T &x) {
        Location loc = x.base.base.loc;
        uint32_t line, column;
        if (compiler_options.emit_debug_line_column) {
            debug_get_line_column(loc.first, line, column);
        } else {
            line = loc.first;
            column = 0;
        }
        builder->SetCurrentDebugLocation(
            llvm::DILocation::get(debug_current_scope->getContext(),
                line, column, debug_current_scope));
    }

    template <typename T>
    void debug_emit_function(const T &x, llvm::DISubprogram *&SP) {
        debug_Unit = DBuilder->createFile(
            debug_CU->getFilename(),
            debug_CU->getDirectory());
        llvm::DIScope *FContext = debug_Unit;
        uint32_t line, column;
        if (compiler_options.emit_debug_line_column) {
            debug_get_line_column(x.base.base.loc.first, line, column);
        } else {
            line = 0;
        }
        std::string fn_debug_name = x.m_name;
        llvm::DIBasicType *return_type_info = nullptr;
        if constexpr (std::is_same_v<T, ASR::Function_t>){
            if(x.m_return_var != nullptr) {
                std::string type_name; uint32_t type_size, type_encoding;
                get_type_debug_info(ASRUtils::expr_type(x.m_return_var),
                    type_name, type_size, type_encoding);
                return_type_info = DBuilder->createBasicType(type_name,
                    type_size, type_encoding);
            }
        } else if constexpr (std::is_same_v<T, ASR::Program_t>) {
            return_type_info = DBuilder->createBasicType("integer", 32,
                llvm::dwarf::DW_ATE_signed);
        }
        llvm::DISubroutineType *return_type = DBuilder->createSubroutineType(
            DBuilder->getOrCreateTypeArray(return_type_info));
        SP = DBuilder->createFunction(
            FContext, fn_debug_name, llvm::StringRef(), debug_Unit,
            line, return_type, 0, // TODO: ScopeLine
            llvm::DINode::FlagPrototyped,
            llvm::DISubprogram::SPFlagDefinition);
        debug_current_scope = SP;
    }

    inline bool verify_dimensions_t(ASR::dimension_t* m_dims, int n_dims) {
        if( n_dims <= 0 ) {
            return false;
        }
        bool is_ok = true;
        for( int r = 0; r < n_dims; r++ ) {
            if( m_dims[r].m_length == nullptr ) {
                is_ok = false;
                break;
            }
        }
        return is_ok;
    }

    void fill_array_details(llvm::Value* arr, llvm::Type* llvm_data_type,
                            ASR::dimension_t* m_dims, int n_dims, bool is_data_only=false,
                            bool reserve_data_memory=true) {
        std::vector<std::pair<llvm::Value*, llvm::Value*>> llvm_dims;
        int64_t ptr_loads_copy = ptr_loads;
        ptr_loads = 2;
        for( int r = 0; r < n_dims; r++ ) {
            ASR::dimension_t m_dim = m_dims[r];
            LCOMPILERS_ASSERT(m_dim.m_start != nullptr);
            visit_expr(*(m_dim.m_start));
            llvm::Value* start = tmp;
            LCOMPILERS_ASSERT(m_dim.m_length != nullptr);
            visit_expr(*(m_dim.m_length));
            llvm::Value* end = tmp;
            llvm_dims.push_back(std::make_pair(start, end));
        }
        ptr_loads = ptr_loads_copy;
        if( is_data_only ) {
            if( !ASRUtils::is_fixed_size_array(m_dims, n_dims) ) {
                llvm::Value* const_1 = llvm::ConstantInt::get(context, llvm::APInt(32, 1));
                llvm::Value* prod = const_1;
                for( int r = 0; r < n_dims; r++ ) {
                    llvm::Value* dim_size = llvm_dims[r].second;
                    prod = builder->CreateMul(prod, dim_size);
                }
                llvm::Value* arr_first = nullptr;
                if( !compiler_options.stack_arrays ) {
                    llvm::DataLayout data_layout(module->getDataLayout());
                    uint64_t size = data_layout.getTypeAllocSize(llvm_data_type);
                    prod = builder->CreateMul(prod,
                        llvm::ConstantInt::get(context, llvm::APInt(32, size)));
                    llvm::Value* arr_first_i8 = LLVMArrUtils::lfortran_malloc(
                        context, *module, *builder, prod);
                    heap_arrays.push_back(arr_first_i8);
                    arr_first = builder->CreateBitCast(
                        arr_first_i8, llvm_data_type->getPointerTo());
                } else {
                    arr_first = llvm_utils->CreateAlloca(*builder, llvm_data_type, prod);
                    builder->CreateStore(arr_first, arr);
                }
            }
        } else {
            arr_descr->fill_array_details(arr, llvm_data_type, n_dims,
                llvm_dims, module.get(), reserve_data_memory);
        }
    }

    /*
        This function fills the descriptor
        (pointer to the first element, offset and descriptor of each dimension)
        of the array which are allocated memory in heap.
    */
    inline void fill_malloc_array_details(llvm::Value* arr, llvm::Type* arr_type, llvm::Type* llvm_data_type,
                                          ASR::ttype_t* asr_type,
                                          ASR::dimension_t* m_dims, int n_dims,
                                          ASR::expr_t* string_len_to_allocate,
                                          bool realloc=false) {
        std::vector<std::pair<llvm::Value*, llvm::Value*>> llvm_dims;
        int ptr_loads_copy = ptr_loads;
        ptr_loads = 2;
        for( int r = 0; r < n_dims; r++ ) {
            ASR::dimension_t m_dim = m_dims[r];
            visit_expr_wrapper(m_dim.m_start, true);
            llvm::Value* start = tmp;
            visit_expr_wrapper(m_dim.m_length, true);
            llvm::Value* end = tmp;
            llvm_dims.push_back(std::make_pair(start, end));
        }
        llvm::Value* string_len{};
        if(string_len_to_allocate){
            visit_expr(*string_len_to_allocate);
            string_len = tmp;
            tmp = nullptr;
        }
        ptr_loads = ptr_loads_copy;
        arr_descr->fill_malloc_array_details(arr, arr_type, llvm_data_type,
            asr_type, n_dims, llvm_dims, string_len, module.get(), realloc);
    }


    std::pair<llvm::Value*, llvm::Value*> get_string_data_and_length(ASR::expr_t* str_expr){
        LCOMPILERS_ASSERT(ASR::is_a<ASR::String_t>(*
            ASRUtils::extract_type(ASRUtils::expr_type(str_expr))))

        // Evaluate Expression
        ASR::String_t* str = ASR::down_cast<ASR::String_t>(
            ASRUtils::extract_type(expr_type(str_expr)));
        this->visit_expr_load_wrapper(str_expr, 0);
        std::pair<llvm::Value*, llvm::Value*> data_and_length;
        switch (str->m_physical_type)
        {
            case ASR::DescriptorString:{
                //Set data
                data_and_length.first = builder->CreateLoad(
                    character_type,
                    llvm_utils->create_gep2(string_descriptor, tmp, 0)); 

                // Set length (Length could be explicit OR implicit)
                if(str->m_len && ASR::is_a<ASR::IntegerConstant_t>(*str->m_len)){ // Explicit-Constant Length
                    ASR::IntegerConstant_t* len = ASR::down_cast<ASR::IntegerConstant_t>(str->m_len);
                    llvm::Value* len_value = llvm::ConstantInt::get(context, llvm::APInt(64, len->m_n));
                    data_and_length.second = len_value;
                } else { // Implicit Length
                    data_and_length.second = builder->CreateLoad(
                        llvm::Type::getInt64Ty(context),
                        llvm_utils->create_gep2(string_descriptor, tmp, 1)); 
                }
                break;
            }
            case ASR::CChar:{

                llvm::Value* char_ptr = builder->CreateAlloca(llvm::Type::getInt8Ty(context));
                data_and_length.first = builder->CreateStore(tmp, char_ptr);
                data_and_length.second = llvm::ConstantInt::get(context, llvm::APInt(64, 1));
                break;
            }
            default:
                throw LCompilersException("Unsupported string physical type.");
        }
        return data_and_length;
    }





    /*
        * Returns length of the string in an array of strings.
    */
    llvm::Value* get_string_length_in_array(ASR::expr_t* expr, llvm::Value* str){
        LCOMPILERS_ASSERT(llvm_utils->is_proper_array_of_strings_llvm_var(expr_type(expr), str))
        ASR::String_t* str_type = ASRUtils::get_string_type(expr_type(expr));
        switch(ASRUtils::extract_physical_type(expr_type(expr))){
            case ASR::DescriptorArray : {
                switch(str_type->m_physical_type){
                    case ASR::DescriptorString : {
                        llvm::Value* temp{};
                        temp = arr_descr->get_pointer_to_data(expr, ASRUtils::expr_type(expr), str, module.get());
                        temp = builder->CreateLoad(
                            llvm_utils->get_el_type(expr, ASRUtils::extract_type(expr_type(expr)), module.get())->getPointerTo(),
                            temp);
                        return builder->CreateLoad(llvm::Type::getInt64Ty(context),
                                llvm_utils->create_gep2(string_descriptor, temp, 1));
                    }
                    default:
                        throw LCompilersException("Unhandled string physicalType");
                }
            }
            case ASR::PointerToDataArray:{
                switch(str_type->m_physical_type){
                    case ASR::DescriptorString : {
                        return builder->CreateLoad(llvm::Type::getInt64Ty(context),
                                llvm_utils->create_gep2(string_descriptor, str, 1));
                    }
                    default:
                        throw LCompilersException("Unhandled string physicalType");

                }
            default:
                throw LCompilersException("Unhandled Array Physical type");
            }

        }
    }


    /*
     * Returns length of the string (int64)
     * It handles string and array of strings.
    */
    llvm::Value* get_string_length(ASR::expr_t* str_expr){
        ASR::ttype_t* exp_type = ASRUtils::expr_type(str_expr);
        LCOMPILERS_ASSERT(ASR::is_a<ASR::String_t>(*
            ASRUtils::extract_type(exp_type)))

        ASR::String_t* str = ASR::down_cast<ASR::String_t>(
            ASRUtils::extract_type(exp_type));


        switch (str->m_physical_type)
        {
            case ASR::DescriptorString:{
                // Set length Length could be explicit
                if(str->m_len && ASR::is_a<ASR::IntegerConstant_t>(*str->m_len)){ // Explicit-Constant Length
                    ASR::IntegerConstant_t* len = ASR::down_cast<ASR::IntegerConstant_t>(str->m_len);
                    llvm::Value* len_value = llvm::ConstantInt::get(context, llvm::APInt(64, len->m_n));
                    return len_value;
                } else { // Implicit Length
                    if(ASRUtils::is_array(exp_type)){
                        visit_expr_load_wrapper(str_expr, ASRUtils::is_allocatable_or_pointer(exp_type) ? 1 : 0);
                        return get_string_length_in_array(str_expr, tmp);
                    } else {
                        visit_expr_load_wrapper(str_expr, 0);
                        LCOMPILERS_ASSERT(llvm_utils->is_proper_string_llvm_variable(str, tmp))
                        return builder->CreateLoad(
                            llvm::Type::getInt64Ty(context),
                            llvm_utils->create_gep2(string_descriptor, tmp, 1)); 
                    }
                }
            }    
            case ASR::CChar:
                return llvm::ConstantInt::get(context, llvm::APInt(64, 1));
            default:
                throw LCompilersException("Unsupported string physical type.");
        }

    }
    
    llvm::Value* get_string_data(ASR::expr_t* str_expr){
        LCOMPILERS_ASSERT(ASR::is_a<ASR::String_t>(*
            ASRUtils::extract_type(ASRUtils::expr_type(str_expr))))

        // Evaluate Expression
        ASR::String_t* str = ASR::down_cast<ASR::String_t>(
            ASRUtils::extract_type(expr_type(str_expr)));
        int ptr_loads_copy = ptr_loads; 
        ptr_loads = 0;
        visit_expr(*str_expr);
        ptr_loads = ptr_loads_copy;
        switch (str->m_physical_type)
        {
            case ASR::DescriptorString:{
                return builder->CreateLoad(
                    character_type,
                    llvm_utils->create_gep2(string_descriptor, tmp, 0)); 
            }    
            case ASR::CChar:{
                llvm::Value* char_ptr = builder->CreateAlloca(llvm::Type::getInt8Ty(context));
                return builder->CreateStore(tmp, char_ptr);
            }
            default:
                throw LCompilersException("Unsupported string physical type.");
        }
        return nullptr;
    }

    void setup_string_length(llvm::Value* str,
        ASR::String_t* str_type, ASR::expr_t* len){
        LCOMPILERS_ASSERT(str)
        switch(str_type->m_len_kind){
            case ASR::ExpressionLength:{
                LCOMPILERS_ASSERT(len);
                int ptr_load_cpy = ptr_loads;ptr_loads = 1;
                visit_expr(*len);
                ptr_loads = ptr_load_cpy;
                tmp = llvm_utils->convert_kind(tmp, llvm::Type::getInt64Ty(context));
                llvm::Value* len_ptr = llvm_utils->get_string_length(str_type, str, true);
                builder->CreateStore(tmp, len_ptr);
                tmp = nullptr;
                break;
            }
            case ASR::AssumedLength: 
                LCOMPILERS_ASSERT_MSG(false,
                    "Shouldn't define assumed length string variable (They're only arguments) ")
                break;
            case ASR::DeferredLength: 
                // Do nothing, deferred length strings doesn't have information to set it up with.
                break;
            default:
                throw LCompilersException("Unhandled string length kind");
                break;
        }
    }
    /*
        Setup a string variable declaration.
        Sets up string's information (data and length)
        based on ASR::String node details.
        - Everything is set to its inital value at first (nullptr, 0).
        - Length is set, if not deferred.
        - Memory gets allocated, if not allocatable.
    */
    void setup_string(llvm::Value* str, ASR::ttype_t* type){
        if(ASRUtils::is_descriptorString(type)){
            builder->CreateStore(llvm::Constant::getNullValue(string_descriptor), str);
            ASR::String_t *t = down_cast<ASR::String_t>(ASRUtils::extract_type(type));
            setup_string_length(str, t, t->m_len);
            // Handle Memory
            if(!ASRUtils::is_allocatable_or_pointer(type)){
                llvm_utils->initialize_string_stack(t->m_physical_type, str, llvm_utils->get_string_length(t, str));
            }
        } else {
            throw LCompilersException("Unhandled string physicalType");
        }
    }

    /*
        *Creates the string (based on stringPhysicalType) used for arrays of strings.
        *This sets the initial state of the array's string. 
        - Creates a single string (based on the physical type).
        - If not allocatable, set data member with (array_size * string_length)
        - If not deferred length, Set length member.
        It doesn't handle the array itself, only the string data it contains.
    */
    llvm::Value* create_and_setup_string_for_array(ASR::ttype_t* var_type, llvm::Value* array_size, bool stack_allocation=false, std::string name = "") {
        LCOMPILERS_ASSERT(ASRUtils::is_array(var_type) && ASRUtils::is_character(*var_type))    
        ASR::String_t* str = ASR::down_cast<ASR::String_t>(ASRUtils::extract_type(var_type));

        if(str->m_physical_type == ASR::DescriptorString){
            llvm::Value* str_desc = llvm_utils->create_empty_string_descriptor(name); // StringDesc with initial state.
            // Setup Length
            setup_string_length(str_desc, str, str->m_len);
            // Setup memory
            if(!ASRUtils::is_allocatable_or_pointer(var_type) /*Length is Fixed*/){
                LCOMPILERS_ASSERT(str->m_len_kind == ASR::ExpressionLength)
                LCOMPILERS_ASSERT( array_size )
                LCOMPILERS_ASSERT( str->m_len )

                tmp = nullptr;
                visit_expr_load_wrapper(str->m_len, 1);
                llvm::Value* str_len = tmp;

                // Arrays are allocated on heap by default.
                stack_allocation ? 
                    llvm_utils->set_array_of_strings_memory_on_stack(str, str_desc, str_len, array_size)
                    :llvm_utils->set_array_of_strings_memory_on_heap(str, str_desc, str_len, array_size);
            }
            return str_desc;
        } else {
            throw LCompilersException("Unhandled string physicalType");
        }
        return nullptr;
    }
    /*
    * Dispatches the required function from runtime library to
    * perform the specified binary operation.
    *
    * @param left_arg llvm::Value* The left argument of the binary operator.
    * @param right_arg llvm::Value* The right argument of the binary operator.
    * @param runtime_func_name std::string The name of the function to be dispatched
    *                                      from runtime library.
    * @returns llvm::Value* The result of the operation.
    *
    * Note
    * ====
    *
    * Internally the call to this function gets transformed into a runtime call:
    * void _lfortran_complex_add(complex* a, complex* b, complex *result)
    *
    * As of now the following values for func_name are supported,
    *
    * _lfortran_complex_add
    * _lfortran_complex_sub
    * _lfortran_complex_div
    * _lfortran_complex_mul
    */
    llvm::Value* lfortran_complex_bin_op(llvm::Value* left_arg, llvm::Value* right_arg,
                                         std::string runtime_func_name,
                                         llvm::Type* complex_type=nullptr)
    {
        if( complex_type == nullptr ) {
            complex_type = complex_type_4;
        }
        llvm::Function *fn = module->getFunction(runtime_func_name);
        if (!fn) {
            llvm::FunctionType *function_type = llvm::FunctionType::get(
                    llvm::Type::getVoidTy(context), {
                        complex_type->getPointerTo(),
                        complex_type->getPointerTo(),
                        complex_type->getPointerTo()
                    }, false);
            fn = llvm::Function::Create(function_type,
                    llvm::Function::ExternalLinkage, runtime_func_name, *module);
        }

        llvm::AllocaInst *pleft_arg = llvm_utils->CreateAlloca(*builder, complex_type);

        builder->CreateStore(left_arg, pleft_arg);
        llvm::AllocaInst *pright_arg = llvm_utils->CreateAlloca(*builder, complex_type);
        builder->CreateStore(right_arg, pright_arg);
        llvm::AllocaInst *presult = llvm_utils->CreateAlloca(*builder, complex_type);
        std::vector<llvm::Value*> args = {pleft_arg, pright_arg, presult};
        builder->CreateCall(fn, args);
        return llvm_utils->CreateLoad(presult);
    }


    llvm::Value* lfortran_strConcat(
        llvm::Value* left_arg, llvm::Value* left_len,
        llvm::Value* right_arg, llvm::Value* right_len)
    {
        std::string runtime_func_name = "_lfortran_strcat";
        llvm::Function *fn = module->getFunction(runtime_func_name);
        if (!fn) {
            llvm::FunctionType *function_type = llvm::FunctionType::get(
                    character_type, {
                        character_type, llvm::Type::getInt64Ty(context),
                        character_type, llvm::Type::getInt64Ty(context)
                    }, false);
            fn = llvm::Function::Create(function_type,
                    llvm::Function::ExternalLinkage, runtime_func_name, *module);
        }
        std::vector<llvm::Value*> args = {left_arg, left_len, right_arg, right_len};
        return builder->CreateCall(fn, args);
    }

    llvm::Value* lfortran_str_cmp(
        llvm::Value* left_arg, llvm::Value* left_arg_len,
        llvm::Value* right_arg, llvm::Value* right_arg_len)
    {
        std::string runtime_func_name = "str_compare";
        llvm::Function *fn = module->getFunction(runtime_func_name);
        if(!fn) {
            llvm::FunctionType *function_type = llvm::FunctionType::get(
                    llvm::Type::getInt32Ty(context), {
                        character_type, llvm::Type::getInt64Ty(context),
                        character_type, llvm::Type::getInt64Ty(context)
                    }, false);
            fn = llvm::Function::Create(function_type,
                    llvm::Function::ExternalLinkage, runtime_func_name, *module);
        }
        std::vector<llvm::Value*> args = {left_arg, left_arg_len, right_arg, right_arg_len};
        return builder->CreateCall(fn, args);
    }

    llvm::Value* lfortran_strrepeat(llvm::Value* left_arg, llvm::Value* right_arg)
    {
        std::string runtime_func_name = "_lfortran_strrepeat";
        llvm::Function *fn = module->getFunction(runtime_func_name);
        if (!fn) {
            llvm::FunctionType *function_type = llvm::FunctionType::get(
                    llvm::Type::getVoidTy(context), {
                        character_type->getPointerTo(),
                        llvm::Type::getInt32Ty(context),
                        character_type->getPointerTo()
                    }, false);
            fn = llvm::Function::Create(function_type,
                    llvm::Function::ExternalLinkage, runtime_func_name, *module);
        }
        llvm::AllocaInst *pleft_arg = llvm_utils->CreateAlloca(*builder, character_type);
        builder->CreateStore(left_arg, pleft_arg);
        llvm::AllocaInst *presult = llvm_utils->CreateAlloca(*builder, character_type);
        std::vector<llvm::Value*> args = {pleft_arg, right_arg, presult};
        builder->CreateCall(fn, args);
        return llvm_utils->CreateLoad(presult);
    }

    llvm::Value* lfortran_str_len(llvm::Value* str, bool use_descriptor=false)
    {
        if (use_descriptor) {
            str = llvm_utils->CreateLoad(arr_descr->get_pointer_to_data(str));
        }
        std::string runtime_func_name = "_lfortran_str_len";
        llvm::Function *fn = module->getFunction(runtime_func_name);
        if (!fn) {
            llvm::FunctionType *function_type = llvm::FunctionType::get(
                    llvm::Type::getInt64Ty(context), {
                        character_type
                    }, false);
            fn = llvm::Function::Create(function_type,
                    llvm::Function::ExternalLinkage, runtime_func_name, *module);
        }
        return builder->CreateCall(fn, {str});
    }

    llvm::Value* lfortran_str_to_int(llvm::Value* str)
    {
        std::string runtime_func_name = "_lfortran_str_to_int";
        llvm::Function *fn = module->getFunction(runtime_func_name);
        if (!fn) {
            llvm::FunctionType *function_type = llvm::FunctionType::get(
                    llvm::Type::getInt32Ty(context), {
                        character_type->getPointerTo()
                    }, false);
            fn = llvm::Function::Create(function_type,
                    llvm::Function::ExternalLinkage, runtime_func_name, *module);
        }
        return builder->CreateCall(fn, {str});
    }

    llvm::Value* lfortran_str_ord(llvm::Value* str)
    {
        std::string runtime_func_name = "_lfortran_str_ord";
        llvm::Function *fn = module->getFunction(runtime_func_name);
        if (!fn) {
            llvm::FunctionType *function_type = llvm::FunctionType::get(
                    llvm::Type::getInt32Ty(context), {
                        character_type->getPointerTo()
                    }, false);
            fn = llvm::Function::Create(function_type,
                    llvm::Function::ExternalLinkage, runtime_func_name, *module);
        }
        return builder->CreateCall(fn, {str});
    }

    llvm::Value* lfortran_str_chr(llvm::Value* str)
    {
        std::string runtime_func_name = "_lfortran_str_chr";
        llvm::Function *fn = module->getFunction(runtime_func_name);
        if (!fn) {
            llvm::FunctionType *function_type = llvm::FunctionType::get(
                    character_type, {
                        llvm::Type::getInt8Ty(context)
                    }, false);
            fn = llvm::Function::Create(function_type,
                    llvm::Function::ExternalLinkage, runtime_func_name, *module);
        }
        return builder->CreateCall(fn, {str});
    }

    llvm::Value* lfortran_str_item(llvm::Value* str, llvm::Value* str_len, llvm::Value* idx1)
    {
        std::string runtime_func_name = "_lfortran_str_item";
        llvm::Function *fn = module->getFunction(runtime_func_name);
        if (!fn) {
            llvm::FunctionType *function_type = llvm::FunctionType::get(
                    character_type, 
                    {
                        character_type /*str*/, 
                        llvm::Type::getInt64Ty(context) /*str_len*/,
                        llvm::Type::getInt64Ty(context) /*idx*/
                    },
                    false);
            fn = llvm::Function::Create(function_type,
                    llvm::Function::ExternalLinkage, runtime_func_name, *module);
        }
        idx1 = builder->CreateSExt(idx1, llvm::Type::getInt64Ty(context));
        return builder->CreateCall(fn, {str, str_len, idx1});
    }

    llvm::Value* lfortran_str_slice(
        llvm::Value* str, llvm::Value* str_len,
        llvm::Value* idx1, llvm::Value* idx2,
        llvm::Value* step, llvm::Value* left_present, llvm::Value* right_present)
    {
        std::string runtime_func_name = "_lfortran_str_slice";
        llvm::Function *fn = module->getFunction(runtime_func_name);
        if (!fn) {
            llvm::FunctionType *function_type = llvm::FunctionType::get(
                    character_type, {
                        character_type, llvm::Type::getInt64Ty(context),
                        llvm::Type::getInt64Ty(context),
                        llvm::Type::getInt64Ty(context), llvm::Type::getInt64Ty(context),
                        llvm::Type::getInt1Ty(context), llvm::Type::getInt1Ty(context)
                    }, false);
            fn = llvm::Function::Create(function_type,
                    llvm::Function::ExternalLinkage, runtime_func_name, *module);
        }
        // Make sure they're int64 integers
        idx1 = llvm_utils->convert_kind(idx1, llvm::Type::getInt64Ty(context));
        idx2 = llvm_utils->convert_kind(idx2, llvm::Type::getInt64Ty(context));
        step = llvm_utils->convert_kind(step, llvm::Type::getInt64Ty(context));
        return builder->CreateCall(fn, {str, str_len, idx1, idx2, step, left_present, right_present});
    }

    // Specific For Fortran Strings
    llvm::Value* lfortran_str_slice_fortran(
        llvm::Value* str,
        llvm::Value* start, llvm::Value* end){
        std::string runtime_func_name = "_lfortran_str_slice_fortran";
        llvm::Function *fn = module->getFunction(runtime_func_name);
        if (!fn) {
            llvm::FunctionType *function_type = llvm::FunctionType::get(
                    character_type, {
                        character_type,
                        llvm::Type::getInt64Ty(context), // Start
                        llvm::Type::getInt64Ty(context) // End
                    }, false);
            fn = llvm::Function::Create(function_type,
                    llvm::Function::ExternalLinkage, runtime_func_name, *module);
        }
        // Make sure they're int64 integers
        start = llvm_utils->convert_kind(start, llvm::Type::getInt64Ty(context));
        end = llvm_utils->convert_kind(end, llvm::Type::getInt64Ty(context));
        return builder->CreateCall(fn, {str, start, end});
    }



    llvm::Value* lfortran_type_to_str(llvm::Value* arg, llvm::Type* value_type, std::string type, int value_kind) {
        std::string func_name = "_lfortran_" + type + "_to_str" + std::to_string(value_kind);
         llvm::Function *fn = module->getFunction(func_name);
         if(!fn) {
            llvm::FunctionType *function_type = llvm::FunctionType::get(
                 character_type, {
                     value_type
                 }, false);
            fn = llvm::Function::Create(function_type,
                     llvm::Function::ExternalLinkage, func_name, *module);
         }
         llvm::Value* res = builder->CreateCall(fn, {arg});
         return res;
    }

    // This function is called as:
    // float complex_re(complex a)
    // And it extracts the real part of the complex number
    llvm::Value *complex_re(llvm::Value *c, llvm::Type* complex_type=nullptr) {
        if( complex_type == nullptr ) {
            complex_type = complex_type_4;
        }
        if( c->getType()->isPointerTy() ) {
            c = llvm_utils->CreateLoad(c);
        }
        llvm::AllocaInst *pc = llvm_utils->CreateAlloca(*builder, complex_type);
        builder->CreateStore(c, pc);
        std::vector<llvm::Value *> idx = {
            llvm::ConstantInt::get(context, llvm::APInt(32, 0)),
            llvm::ConstantInt::get(context, llvm::APInt(32, 0))};
        llvm::Value *pim = llvm_utils->CreateGEP2(complex_type, pc, idx);
        if (complex_type == complex_type_4) {
            return llvm_utils->CreateLoad2(llvm::Type::getFloatTy(context), pim);
        } else {
            return llvm_utils->CreateLoad2(llvm::Type::getDoubleTy(context), pim);
        }
    }

    llvm::Value *complex_im(llvm::Value *c, llvm::Type* complex_type=nullptr) {
        if( complex_type == nullptr ) {
            complex_type = complex_type_4;
        }
        llvm::AllocaInst *pc = llvm_utils->CreateAlloca(*builder, complex_type);
        builder->CreateStore(c, pc);
        std::vector<llvm::Value *> idx = {
            llvm::ConstantInt::get(context, llvm::APInt(32, 0)),
            llvm::ConstantInt::get(context, llvm::APInt(32, 1))};
        llvm::Value *pim = llvm_utils->CreateGEP2(complex_type, pc, idx);
        if (complex_type == complex_type_4) {
            return llvm_utils->CreateLoad2(llvm::Type::getFloatTy(context), pim);
        } else {
            return llvm_utils->CreateLoad2(llvm::Type::getDoubleTy(context), pim);
        }
    }

    llvm::Value *complex_from_floats(llvm::Value *re, llvm::Value *im,
                                     llvm::Type* complex_type=nullptr) {
        if( complex_type == nullptr ) {
            complex_type = complex_type_4;
        }
        llvm::AllocaInst *pres = llvm_utils->CreateAlloca(*builder, complex_type);
        std::vector<llvm::Value *> idx1 = {
            llvm::ConstantInt::get(context, llvm::APInt(32, 0)),
            llvm::ConstantInt::get(context, llvm::APInt(32, 0))};
        std::vector<llvm::Value *> idx2 = {
            llvm::ConstantInt::get(context, llvm::APInt(32, 0)),
            llvm::ConstantInt::get(context, llvm::APInt(32, 1))};
        llvm::Value *pre = llvm_utils->CreateGEP2(complex_type, pres, idx1);
        llvm::Value *pim = llvm_utils->CreateGEP2(complex_type, pres, idx2);
        builder->CreateStore(re, pre);
        builder->CreateStore(im, pim);
        return llvm_utils->CreateLoad2(complex_type, pres);
    }

    llvm::Value *nested_struct_rd(std::vector<llvm::Value*> vals,
            llvm::StructType* rd) {
        llvm::AllocaInst *pres = llvm_utils->CreateAlloca(*builder, rd);
        llvm::Value *pim = llvm_utils->CreateGEP2(rd, pres, vals);
        return llvm_utils->CreateLoad(pim);
    }

    /**
     * @brief This function generates the
     * @detail This is converted to
     *
     *     float lfortran_KEY(float *x)
     *
     *   Where KEY can be any of the supported intrinsics; this is then
     *   transformed into a runtime call:
     *
     *     void _lfortran_KEY(float x, float *result)
     */
    llvm::Value* lfortran_intrinsic(llvm::Function *fn, llvm::Value* pa, int a_kind)
    {
        llvm::Type *presult_type = llvm_utils->getFPType(a_kind);
        llvm::AllocaInst *presult = llvm_utils->CreateAlloca(*builder, presult_type);
        llvm::Value *a = llvm_utils->CreateLoad(pa);
        std::vector<llvm::Value*> args = {a, presult};
        builder->CreateCall(fn, args);
        return llvm_utils->CreateLoad(presult);
    }

    llvm::Type* get_llvm_struct_data_type(ASR::Struct_t* st, bool is_pointer) {
        std::string struct_name = (std::string)st->m_name;
        if (struct_name == "~unlimited_polymorphic_type") {
            if (is_pointer) {
                return llvm::Type::getVoidTy(context)->getPointerTo();
            } else {
                return llvm::Type::getVoidTy(context);
            }
        } else {
            return llvm_utils->getStructType(st, module.get(), is_pointer);
        }
    }

    void visit_TranslationUnit(const ASR::TranslationUnit_t &x) {
        module = std::make_unique<llvm::Module>("LFortran", context);
        module->setDataLayout("");
        llvm_utils->set_module(module.get());

        if (compiler_options.emit_debug_info) {
            DBuilder = std::make_unique<llvm::DIBuilder>(*module);
            debug_CU = DBuilder->createCompileUnit(
                llvm::dwarf::DW_LANG_C, DBuilder->createFile(infile, "."),
                "LPython Compiler", false, "", 0);
        }

        // All loose statements must be converted to a function, so the items
        // must be empty:
        LCOMPILERS_ASSERT(x.n_items == 0);

        // Define LLVM types that we might need
        // Complex type is represented as an identified struct in LLVM
        // %complex = type { float, float }
        complex_type_4 = llvm_utils->complex_type_4;
        complex_type_8 = llvm_utils->complex_type_8;
        complex_type_4_ptr = llvm_utils->complex_type_4_ptr;
        complex_type_8_ptr = llvm_utils->complex_type_8_ptr;
        character_type = llvm_utils->character_type;
        string_descriptor = llvm_utils->string_descriptor;
        list_type = llvm::Type::getInt8Ty(context)->getPointerTo();

        llvm::Type* bound_arg = static_cast<llvm::Type*>(arr_descr->get_dimension_descriptor_type(true));
        fname2arg_type["lbound"] = std::make_pair(bound_arg, bound_arg->getPointerTo());
        fname2arg_type["ubound"] = std::make_pair(bound_arg, bound_arg->getPointerTo());

        // Process Variables first:
        for (auto &item : x.m_symtab->get_scope()) {
            if (is_a<ASR::Variable_t>(*item.second) ||
                is_a<ASR::Enum_t>(*item.second)) {
                visit_symbol(*item.second);
            }
        }

        prototype_only = false;
        for (auto &item : x.m_symtab->get_scope()) {
            if (is_a<ASR::Module_t>(*item.second) &&
                item.first.find("lfortran_intrinsic_optimization") != std::string::npos) {
                ASR::Module_t* mod = ASR::down_cast<ASR::Module_t>(item.second);
                for( auto &moditem: mod->m_symtab->get_scope() ) {
                    ASR::symbol_t* sym = ASRUtils::symbol_get_past_external(moditem.second);
                    if (is_a<ASR::Function_t>(*sym)) {
                        visit_Function(*ASR::down_cast<ASR::Function_t>(sym));
                    }
                }
            }
        }

        prototype_only = true;
        // Generate function prototypes
        for (auto &item : x.m_symtab->get_scope()) {
            if (is_a<ASR::Function_t>(*item.second)) {
                visit_Function(*ASR::down_cast<ASR::Function_t>(item.second));
            }
        }
        prototype_only = false;

        // TODO: handle dependencies across modules and main program

        // Then do all the modules in the right order
        std::vector<std::string> build_order
            = determine_module_dependencies(x);
        for (auto &item : build_order) {
            if (!item.compare("_lcompilers_mlir_gpu_offloading")) continue;
            LCOMPILERS_ASSERT(x.m_symtab->get_symbol(item)
                != nullptr);
            ASR::symbol_t *mod = x.m_symtab->get_symbol(item);
            visit_symbol(*mod);
        }

        // Then do all the procedures
        for (auto &item : x.m_symtab->get_scope()) {
            if( ASR::is_a<ASR::Function_t>(*item.second) ) {
                visit_symbol(*item.second);
            }
        }

        // Then the main program
        for (auto &item : x.m_symtab->get_scope()) {
            if (is_a<ASR::Program_t>(*item.second)) {
                visit_symbol(*item.second);
            }
        }
    }

    template <typename T>
    void visit_AllocateUtil(const T& x, ASR::expr_t* m_stat, bool realloc, ASR::expr_t* m_source = nullptr) {
        for( size_t i = 0; i < x.n_args; i++ ) {
            ASR::alloc_arg_t curr_arg = x.m_args[i];
            ASR::expr_t* tmp_expr = x.m_args[i].m_a;
            LCOMPILERS_ASSERT(
                ASRUtils::is_allocatable(tmp_expr)||
                ASRUtils::is_pointer(expr_type(tmp_expr)));
            visit_expr_load_wrapper(tmp_expr, 0);
            llvm::Value* x_arr = tmp;
            ASR::ttype_t* curr_arg_m_a_type = ASRUtils::type_get_past_pointer(
                ASRUtils::type_get_past_allocatable(
                ASRUtils::expr_type(tmp_expr)));
            size_t n_dims = ASRUtils::extract_n_dims_from_ttype(curr_arg_m_a_type);
            curr_arg_m_a_type = ASRUtils::type_get_past_array(curr_arg_m_a_type);
            if( n_dims == 0 ) {
                if (ASRUtils::is_character(*curr_arg_m_a_type)) {
                    ASR::String_t* str = ASR::down_cast<ASR::String_t>(ASRUtils::extract_type(ASRUtils::expr_type(tmp_expr)));
                    llvm::Value* amount_to_allocate{};
                    if(curr_arg.m_len_expr){ // Visit desired length to be allocated.
                        visit_expr_load_wrapper(curr_arg.m_len_expr, 1);
                        amount_to_allocate = tmp;
                    } else {
                        amount_to_allocate = nullptr;
                    }
                    llvm_utils->allocate_allocatable_string(str, x_arr, amount_to_allocate);
                } else if(ASR::is_a<ASR::Integer_t>(*curr_arg_m_a_type) ||
                          ASR::is_a<ASR::Real_t>(*curr_arg_m_a_type) ||
                          ASR::is_a<ASR::Complex_t>(*curr_arg_m_a_type) ||
                          ASR::is_a<ASR::Logical_t>(*curr_arg_m_a_type)) {
                    llvm::Value* malloc_size = SizeOfTypeUtil(curr_arg.m_a, curr_arg_m_a_type, llvm_utils->getIntType(4),
                    ASRUtils::TYPE(ASR::make_Integer_t(al, x.base.base.loc, 4)));
                    llvm::Value* malloc_ptr = LLVMArrUtils::lfortran_malloc(
                        context, *module, *builder, malloc_size);
                    builder->CreateMemSet(malloc_ptr, llvm::ConstantInt::get(context, llvm::APInt(8, 0)), malloc_size, llvm::MaybeAlign());
                    llvm::Type* llvm_arg_type = llvm_utils->get_type_from_ttype_t_util(curr_arg.m_a, curr_arg_m_a_type, module.get());
                    builder->CreateStore(builder->CreateBitCast(
                        malloc_ptr, llvm_arg_type->getPointerTo()), x_arr);
                } else if (ASR::is_a<ASR::StructType_t>(*curr_arg_m_a_type)) {
                    // TODO: Implement source argument when m_source is class
                    bool m_source_is_class = m_source && ASRUtils::is_class_type(ASRUtils::type_get_past_allocatable(ASRUtils::expr_type(m_source)));
                    if (ASR::down_cast<ASR::StructType_t>(curr_arg_m_a_type)->m_is_cstruct) {
                        llvm::Value* malloc_size = SizeOfTypeUtil(curr_arg.m_a, curr_arg_m_a_type, llvm_utils->getIntType(4),
                        ASRUtils::TYPE(ASR::make_Integer_t(al, x.base.base.loc, 4)));
                        llvm::Value* malloc_ptr = LLVMArrUtils::lfortran_malloc(
                            context, *module, *builder, malloc_size);
                        builder->CreateMemSet(malloc_ptr, llvm::ConstantInt::get(context, llvm::APInt(8, 0)), malloc_size, llvm::MaybeAlign());
                        llvm::Type* llvm_arg_type = llvm_utils->get_type_from_ttype_t_util(curr_arg.m_a, curr_arg_m_a_type, module.get());
                        builder->CreateStore(builder->CreateBitCast(
                            malloc_ptr, llvm_arg_type->getPointerTo()), x_arr);

                        x_arr = llvm_utils->CreateLoad2(llvm_arg_type->getPointerTo(), x_arr);
                        allocate_array_members_of_struct(
                            ASR::down_cast<ASR::Struct_t>(
                                ASRUtils::symbol_get_past_external(ASRUtils::get_struct_sym_from_struct_expr(tmp_expr))),
                            x_arr,
                            curr_arg_m_a_type);
                        if (m_source && !m_source_is_class) {
                            int64_t ptr_loads_copy = ptr_loads;
                            if (ASRUtils::is_allocatable(m_source)) {
                                ptr_loads = 1;
                            } else {
                                ptr_loads = 0;
                            }
                            this->visit_expr(*m_source);
                            ptr_loads = ptr_loads_copy;

                            llvm_utils->deepcopy(m_source, tmp, x_arr, ASRUtils::expr_type(m_source), curr_arg_m_a_type, module.get(), name2memidx);
                        }
                    } else {
                        ASR::ttype_t* dest_asr_type = curr_arg.m_type;
                        ASR::symbol_t* dest_class_sym = nullptr;
                        llvm::Value* malloc_size = nullptr;
                        // If no type specified then use curr_arg_m_a_type as default
                        if (m_source && !m_source_is_class) {
                            if(ASRUtils::is_character(*ASRUtils::expr_type(m_source))){ //hackish (clean dest_type + src_type passed to deepcopy)
                                ASRUtils::ASRBuilder b(al, x.base.base.loc) ;
                                dest_asr_type = b.allocatable(b.String(nullptr, ASR::DeferredLength));
                            } else {
                                dest_asr_type = ASRUtils::type_get_past_allocatable(ASRUtils::expr_type(m_source));
                            }
                            dest_class_sym = ASRUtils::symbol_get_past_external(ASRUtils::get_struct_sym_from_struct_expr(m_source));
                            malloc_size = SizeOfTypeUtil(m_source, dest_asr_type, llvm_utils->getIntType(4),
                                ASRUtils::TYPE(ASR::make_Integer_t(al, x.base.base.loc, 4)));
                        } else if (curr_arg.m_sym_subclass == nullptr) {
                            dest_class_sym = ASRUtils::symbol_get_past_external(ASRUtils::get_struct_sym_from_struct_expr(curr_arg.m_a));
                            dest_asr_type = ASRUtils::make_StructType_t_util(al, curr_arg_m_a_type->base.loc, dest_class_sym);
                            malloc_size = SizeOfTypeUtil(curr_arg.m_a, dest_asr_type, llvm_utils->getIntType(4),
                                ASRUtils::TYPE(ASR::make_Integer_t(al, x.base.base.loc, 4)));
                        } else {                            
                            llvm::Type* llvm_type = nullptr;
                            dest_class_sym = ASRUtils::symbol_get_past_external(curr_arg.m_sym_subclass);
                            ASR::Struct_t* dest_struct = ASR::down_cast<ASR::Struct_t>(dest_class_sym);
                            llvm_type = llvm_utils->getStructType(dest_struct, module.get());
                            llvm::DataLayout data_layout(module->getDataLayout());
                            int64_t type_size = data_layout.getTypeAllocSize(llvm_type);
                            malloc_size = llvm::ConstantInt::get(llvm_utils->getIntType(4), llvm::APInt(
                                ASRUtils::extract_kind_from_ttype_t(ASRUtils::TYPE(
                                    ASR::make_Integer_t(al, x.base.base.loc, 4))) * 8, type_size));
                        }
                        
                        llvm::Value* malloc_ptr = LLVMArrUtils::lfortran_malloc(
                            context, *module, *builder, malloc_size);
                        builder->CreateMemSet(malloc_ptr, llvm::ConstantInt::get(context, llvm::APInt(8, 0)),
                                    malloc_size, llvm::MaybeAlign());

                        // Set class hash in polymorphic struct
                        llvm::Value* class_hash = nullptr;
                        if (dest_class_sym) {
                            class_hash = llvm::ConstantInt::get(llvm_utils->getIntType(8),
                                llvm::APInt(64, get_class_hash(dest_class_sym)));
                        } else {
                            class_hash = llvm::ConstantInt::get(llvm_utils->getIntType(8),
                                llvm::APInt(64, -((int) dest_asr_type->type) -
                                    ASRUtils::extract_kind_from_ttype_t(dest_asr_type), true));
                        }
                        llvm::Type* src_class_type = llvm_utils->get_type_from_ttype_t_util(tmp_expr, curr_arg_m_a_type, module.get());
                        llvm::Value* t = llvm_utils->create_gep2(src_class_type, x_arr, 0);
                        builder->CreateStore(class_hash, t);

                        // Store and bitcast allocated memory into polymorphic struct's struct pointer
                        ASR::Struct_t* src_struct_sym = ASR::down_cast<ASR::Struct_t>(ASRUtils::symbol_get_past_external(ASRUtils::get_struct_sym_from_struct_expr(curr_arg.m_a)));
                        llvm::Type* src_struct_type = get_llvm_struct_data_type(
                            src_struct_sym,
                            true);
                        x_arr = llvm_utils->create_gep2(src_class_type, x_arr, 1);
                        builder->CreateStore(builder->CreateBitCast(
                                        malloc_ptr, src_struct_type), x_arr);

                        // Initialize members
                        llvm::Type* dest_type = nullptr;
                        if (curr_arg.m_sym_subclass) {
                            dest_type = llvm_utils->getStructType(ASR::down_cast<ASR::Struct_t>(dest_class_sym), module.get());
                        } else if (m_source && !m_source_is_class) {
                            dest_type = llvm_utils->get_type_from_ttype_t_util(m_source, dest_asr_type, module.get());
                        } else {
                            dest_type = llvm_utils->get_type_from_ttype_t_util(curr_arg.m_a, dest_asr_type, module.get());
                        }
                        x_arr = llvm_utils->CreateLoad2(src_struct_type, x_arr);
                        x_arr = builder->CreateBitCast(x_arr, dest_type->getPointerTo());

                        if (ASR::is_a<ASR::StructType_t>(*dest_asr_type)) {
                            allocate_array_members_of_struct(ASR::down_cast<ASR::Struct_t>(dest_class_sym), x_arr, dest_asr_type);                        
                        }
                        if (m_source && !m_source_is_class) {
                            int64_t ptr_loads_copy = ptr_loads;
                            if (ASRUtils::is_allocatable(m_source)) {
                                ptr_loads = 1;
                            } else {
                                ptr_loads = 0;
                            }
                            this->visit_expr(*m_source);
                            ptr_loads = ptr_loads_copy;

                            llvm_utils->deepcopy(m_source, tmp, x_arr, dest_asr_type, dest_asr_type, module.get(), name2memidx);
                        }
                    }
                }
            } else {
                ASR::ttype_t* asr_data_type = ASRUtils::duplicate_type_without_dims(al,
                    curr_arg_m_a_type, curr_arg_m_a_type->base.loc);
                llvm::Type* llvm_data_type = llvm_utils->get_type_from_ttype_t_util(curr_arg.m_a, asr_data_type, module.get());
                llvm::Type* type = llvm_utils->get_type_from_ttype_t_util(tmp_expr,
                    ASRUtils::type_get_past_pointer(ASRUtils::type_get_past_allocatable(
                        ASRUtils::expr_type(tmp_expr))), module.get());
                llvm::Value* ptr_val = x_arr;
#if LLVM_VERSION_MAJOR >= 17
                llvm::Type* i8_ptr_ty
                    = llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context));
#else
                llvm::Type* i8_ptr_ty = llvm::Type::getInt8PtrTy(context);
#endif

                if (x_arr && x_arr->getType() == nullptr) {
                    ptr_val = llvm::ConstantPointerNull::get(static_cast<llvm::PointerType*>(i8_ptr_ty));
                }
                llvm_utils->create_if_else(
                    builder->CreateICmpEQ(
                        builder->CreatePtrToInt(llvm_utils->CreateLoad2(type->getPointerTo(), (x_arr && x_arr->getType() != nullptr) ? x_arr : ptr_val), llvm::Type::getInt32Ty(context)),
                        builder->CreatePtrToInt(
                            llvm::ConstantPointerNull::get((x_arr && x_arr->getType() != nullptr) ? x_arr->getType()->getPointerTo() : ptr_val->getType()->getPointerTo()),
                            llvm::Type::getInt32Ty(context))),
                    [&]() {
                        llvm::Value* ptr_;

                            ptr_ = llvm_utils->CreateAlloca(*builder, type);
#if LLVM_VERSION_MAJOR > 16
                            ptr_type[ptr_] = type;
#endif
                            arr_descr->fill_dimension_descriptor(ptr_, n_dims);

                            LLVM::CreateStore(
                                *builder, ptr_, (x_arr && x_arr->getType() != nullptr) ? x_arr : ptr_val);
                    },
                    []() {});
                fill_malloc_array_details((x_arr && x_arr->getType() != nullptr) ? x_arr : ptr_val, 
                                        type, llvm_data_type,
                    expr_type(x.m_args[i].m_a), 
                    curr_arg.m_dims, curr_arg.n_dims, curr_arg.m_len_expr,
                    realloc);
                if( ASR::is_a<ASR::StructType_t>(*ASRUtils::extract_type(ASRUtils::expr_type(tmp_expr))) 
                    && !ASRUtils::is_class_type(ASRUtils::expr_type(tmp_expr)) ) {
                    llvm::Value* x_arr_ = llvm_utils->CreateLoad(x_arr);
#if LLVM_VERSION_MAJOR > 16
                            ptr_type[x_arr_] = type;
#endif
                    allocate_array_members_of_struct_arrays(tmp_expr, x_arr_,
                        ASRUtils::expr_type(tmp_expr));
                }
            }
        }
        if (m_stat) {
            ASR::Variable_t *asr_target = EXPR2VAR(m_stat);
            uint32_t h = get_hash((ASR::asr_t*)asr_target);
            if (llvm_symtab.find(h) != llvm_symtab.end()) {
                llvm::Value *target, *value;
                target = llvm_symtab[h];
                // Store 0 (success) in the stat variable
                ASR::ttype_t* stat_type = ASRUtils::expr_type(m_stat);
                int kind = ASRUtils::extract_kind_from_ttype_t(stat_type);
                value = llvm::ConstantInt::get(context, llvm::APInt(8*kind, 0));
                builder->CreateStore(value, target);
            } else {
                throw CodeGenError("Stat variable in allocate not found in LLVM symtab");
            }
        }
    }

    void visit_Allocate(const ASR::Allocate_t& x) {
        visit_AllocateUtil(x, x.m_stat, false, x.m_source);
    }

    void visit_ReAlloc(const ASR::ReAlloc_t& x) {
        LCOMPILERS_ASSERT(x.n_args == 1);
        handle_allocated(x.m_args[0].m_a);
        llvm::Value* is_allocated = tmp;
        llvm::Value* size = llvm::ConstantInt::get(
            llvm::Type::getInt32Ty(context), llvm::APInt(32, 1));
        int64_t ptr_loads_copy = ptr_loads;
        for( size_t i = 0; i < x.m_args[0].n_dims; i++ ) {
            ptr_loads = 2 - !LLVM::is_llvm_pointer(*
                ASRUtils::expr_type(x.m_args[0].m_dims[i].m_length));
            this->visit_expr_wrapper(x.m_args[0].m_dims[i].m_length, true);
            size = builder->CreateMul(size, tmp);
        }
        ptr_loads = ptr_loads_copy;
        visit_ArraySizeUtil(x.m_args[0].m_a,
            ASRUtils::TYPE(ASR::make_Integer_t(al, x.base.base.loc, 4)));
        llvm::Value* arg_array_size = tmp;
        llvm::Value* realloc_condition = builder->CreateOr(
            builder->CreateNot(is_allocated), builder->CreateAnd(
                is_allocated, builder->CreateICmpNE(size, arg_array_size)));
        // Add more condition if it's array of characters (check lengths)
        if(ASRUtils::is_character(*expr_type(x.m_args[0].m_a))){
            llvm::Value* current_str_len = get_string_length(x.m_args[0].m_a);
            LCOMPILERS_ASSERT(current_str_len->getType() == llvm::Type::getInt64Ty(context));
            visit_expr(*(x.m_args[0].m_len_expr));
            llvm::Value* desired_str_len = llvm_utils->convert_kind(tmp, llvm::Type::getInt64Ty(context));
            tmp = nullptr;
            realloc_condition = builder->CreateOr(realloc_condition, 
                builder->CreateICmpNE(current_str_len, desired_str_len));
        }
        llvm_utils->create_if_else(realloc_condition, [=]() {
            visit_AllocateUtil(x, nullptr, true);
        }, [](){});
    }

    void visit_Nullify(const ASR::Nullify_t& x) {
        for( size_t i = 0; i < x.n_vars; i++ ) {
            ASR::symbol_t* tmp_sym;
            llvm::Value *target;
            if (ASR::is_a<ASR::StructInstanceMember_t>(*x.m_vars[i])) {
                tmp_sym = ASR::down_cast<ASR::StructInstanceMember_t>(x.m_vars[i])->m_m;
                this->visit_expr(*x.m_vars[i]);
                target = tmp;
            } else if (ASR::is_a<ASR::Var_t>(*x.m_vars[i])) {
                tmp_sym = ASR::down_cast<ASR::Var_t>(x.m_vars[i])->m_v;
                tmp_sym = ASRUtils::symbol_get_past_external(tmp_sym);
                std::uint32_t h = get_hash((ASR::asr_t*)tmp_sym);
                target = llvm_symtab[h];
            } else {
                throw CodeGenError("Only StructInstanceMember and Variable are supported Nullify type");
            }

            llvm::Type* tp = llvm_utils->get_type_from_ttype_t_util(x.m_vars[i],
                ASRUtils::type_get_past_pointer(
                ASRUtils::type_get_past_allocatable(
                ASRUtils::symbol_type(tmp_sym))), module.get());

            llvm::Type* dest_type = tp->getPointerTo();
            if (ASR::is_a<ASR::FunctionType_t>(*ASRUtils::symbol_type(tmp_sym)) ||
                ASRUtils::is_class_type(ASRUtils::extract_type(ASRUtils::symbol_type(tmp_sym)))) {
                // functions are pointers in LLVM, so we do not need to get the pointer to it
                // Class type target is already pointer, so we dont need to get the pointer for dest
                dest_type = tp;
            }


            if(ASRUtils::is_array(ASRUtils::expr_type(x.m_vars[i]))){
                llvm::Type* x_m_vars_type = llvm_utils->get_type_from_ttype_t_util(x.m_vars[i],
                    ASRUtils::expr_type(x.m_vars[i]), module.get());
                llvm::Value* target_ = llvm_utils->CreateLoad2(x_m_vars_type, target);
                llvm::Value* data_ptr = arr_descr->get_pointer_to_data(x.m_vars[i], ASRUtils::expr_type(x.m_vars[i]), target_ , module.get());
                builder->CreateStore(
                    llvm::ConstantPointerNull::get(llvm_utils->get_el_type(x.m_vars[i],
                        ASRUtils::extract_type(ASRUtils::expr_type(x.m_vars[i])), module.get())->getPointerTo())
                    , data_ptr);
            } else {
                llvm::Value* np = builder->CreateIntToPtr(
                    llvm::ConstantInt::get(context, llvm::APInt(32, 0)), dest_type);
                builder->CreateStore(np, target);
            }
        }
    }

    inline void call_lfortran_free(llvm::Function* fn, llvm::Type* llvm_data_type) {
        llvm::Value* arr = llvm_utils->CreateLoad2(llvm_data_type->getPointerTo(), arr_descr->get_pointer_to_data(tmp));
        llvm::AllocaInst *arg_arr = llvm_utils->CreateAlloca(*builder, character_type);
        builder->CreateStore(builder->CreateBitCast(arr, character_type), arg_arr);
        std::vector<llvm::Value*> args = {llvm_utils->CreateLoad(arg_arr)};
        builder->CreateCall(fn, args);
        arr_descr->reset_is_allocated_flag(tmp, llvm_data_type);
    }

    llvm::Function* _Deallocate() {
        std::string func_name = "_lfortran_free";
        llvm::Function *free_fn = module->getFunction(func_name);
        if (!free_fn) {
            llvm::FunctionType *function_type = llvm::FunctionType::get(
                    llvm::Type::getVoidTy(context), {
                        character_type
                    }, false);
            free_fn = llvm::Function::Create(function_type,
                    llvm::Function::ExternalLinkage, func_name, *module);
        }
        return free_fn;
    }

    llvm::Function* _Allocate() {
        std::string func_name = "_lfortran_allocate_string";
        llvm::Function *alloc_fun = module->getFunction(func_name);
        if (!alloc_fun) {
            llvm::FunctionType *function_type = llvm::FunctionType::get(
                    llvm::Type::getVoidTy(context), {
                        character_type->getPointerTo(),
                        llvm::Type::getInt64Ty(context),
                        llvm::Type::getInt64Ty(context)->getPointerTo(),
                        llvm::Type::getInt64Ty(context)->getPointerTo()
                    }, false);
            alloc_fun = llvm::Function::Create(function_type,
                    llvm::Function::ExternalLinkage, func_name, *module);
        }
        return alloc_fun;
    }

    template <typename T>
    void visit_Deallocate(const T& x) {
        llvm::Function* free_fn = llvm_utils->_Deallocate();
        for( size_t i = 0; i < x.n_vars; i++ ) {
            ASR::expr_t* tmp_expr = x.m_vars[i];
            ASR::symbol_t* curr_obj = nullptr;
            ASR::abiType abt = ASR::abiType::Source;
            if( ASR::is_a<ASR::Var_t>(*tmp_expr) ) {
                const ASR::Var_t* tmp_var = ASR::down_cast<ASR::Var_t>(tmp_expr);
                curr_obj = tmp_var->m_v;
                ASR::Variable_t *v = ASR::down_cast<ASR::Variable_t>(
                                    symbol_get_past_external(curr_obj));
                int64_t ptr_loads_copy = ptr_loads;
                ptr_loads = 0;
                if (!ASRUtils::is_class_type(ASRUtils::extract_type(v->m_type))) {
                    fetch_var(v);
                } else {
                    uint32_t h = get_hash((ASR::asr_t*)v);
                    tmp = llvm_symtab[h];
                }
                ptr_loads = ptr_loads_copy;
                abt = v->m_abi;
            } else if (ASR::is_a<ASR::StructInstanceMember_t>(*tmp_expr)) {
                ASR::StructInstanceMember_t* sm = ASR::down_cast<ASR::StructInstanceMember_t>(tmp_expr);
                ASR::ttype_t* caller_type = ASRUtils::type_get_past_allocatable(
                        ASRUtils::expr_type(sm->m_v));
                int64_t ptr_loads_copy = ptr_loads;
                if (ASRUtils::is_class_type(ASRUtils::extract_type(caller_type))) {
                    ptr_loads = 0;
                } else {
                    ptr_loads = 1;
                }
                this->visit_expr_wrapper(sm->m_v);
                ptr_loads = ptr_loads_copy;
                llvm::Value* dt = tmp;
                ASR::symbol_t *struct_sym = nullptr;
                llvm::Type* dt_type = llvm_utils->getStructType(
                    ASR::down_cast<ASR::Struct_t>(
                        ASRUtils::symbol_get_past_external(ASRUtils::get_struct_sym_from_struct_expr(sm->m_v))),
                    module.get());
                struct_sym = ASRUtils::symbol_get_past_external(
                    ASRUtils::symbol_get_past_external(ASRUtils::get_struct_sym_from_struct_expr(sm->m_v)));
                if (ASRUtils::is_class_type(ASRUtils::extract_type(caller_type))) {
                    llvm::Type* dt_type_poly = llvm_utils->get_type_from_ttype_t_util(sm->m_v,
                        ASRUtils::type_get_past_pointer(ASRUtils::type_get_past_allocatable(caller_type)),
                        module.get());
                    llvm::Value* dt_ptr = llvm_utils->create_gep2(dt_type_poly, dt, 1);
                    dt = llvm_utils->CreateLoad2(dt_type->getPointerTo(), dt_ptr);
                } else if (ASR::is_a<ASR::StructInstanceMember_t>(*sm->m_v)) {
                    dt = llvm_utils->CreateLoad2(dt_type->getPointerTo(), dt);
                }

                std::string curr_struct = ASRUtils::symbol_name(struct_sym);
                std::string member_name = ASRUtils::symbol_name(ASRUtils::symbol_get_past_external(sm->m_m));
                while( name2memidx[curr_struct].find(member_name) == name2memidx[curr_struct].end() ) {
                    if( dertype2parent.find(curr_struct) == dertype2parent.end() ) {
                        throw CodeGenError(curr_struct + " doesn't have any member named " + member_name,
                                            x.base.base.loc);
                    }
                    dt = llvm_utils->create_gep2(name2dertype[curr_struct], dt, 0);
                    curr_struct = dertype2parent[curr_struct];
                }
                int dt_idx = name2memidx[curr_struct][member_name];
                std::vector<llvm::Value*> idx_vars = {llvm::ConstantInt::get(context, llvm::APInt(32, 0)),
                    llvm::ConstantInt::get(context, llvm::APInt(32, dt_idx))};
                if (dt->getType() != name2dertype[curr_struct]->getPointerTo()) {
                    llvm::Value* dt_val = dt;
                    llvm::Value* alloca_tmp = builder->CreateAlloca(name2dertype[curr_struct]);
                    builder->CreateStore(dt_val, alloca_tmp);
                    dt = alloca_tmp;
                }
                LCOMPILERS_ASSERT(dt->getType()->isPointerTy());
                llvm::Value* dt_1 = builder->CreateGEP(name2dertype[curr_struct], dt, idx_vars);
#if LLVM_VERSION_MAJOR > 16
                llvm::Type *dt_1_type = llvm_utils->get_type_from_ttype_t_util(ASRUtils::get_expr_from_sym(al, sm->m_m),
                    ASRUtils::type_get_past_pointer(ASRUtils::type_get_past_allocatable(
                    ASRUtils::symbol_type(ASRUtils::symbol_get_past_external(sm->m_m)))),
                    module.get());
                ptr_type[dt_1] = dt_1_type;
#endif
                tmp = dt_1;
            } else {
                throw CodeGenError("Cannot deallocate variables in expression " +
                                    ASRUtils::type_to_str_python(ASRUtils::expr_type(tmp_expr)),
                                    tmp_expr->base.loc);
            }
            ASR::ttype_t *cur_type = ASRUtils::expr_type(tmp_expr);
            int dims = ASRUtils::extract_n_dims_from_ttype(cur_type);
            if(ASRUtils::is_character(*cur_type)) { // Handle Strings (array of strings or just string)
                tmp = LLVM::is_llvm_pointer(*cur_type) ?
                    builder->CreateLoad(llvm_utils->get_type_from_ttype_t_util(tmp_expr, cur_type, module.get()), tmp)
                    : tmp ;
                llvm_utils->free_strings(tmp_expr, tmp);
            } else {
                if (dims == 0) {
                    llvm::Type* llvm_data_type;
                    llvm::Value* tmp_ = tmp;
                    if (LLVM::is_llvm_pointer(*cur_type)
                        && !ASRUtils::is_class_type(ASRUtils::extract_type(cur_type))) {
                        tmp = llvm_utils->CreateLoad(tmp);
                    }
                    if (ASRUtils::is_class_type(ASRUtils::extract_type(cur_type))) {
                        // If it is a class type, we need to get the pointer to the struct
                        llvm::Type* class_type = llvm_utils->get_type_from_ttype_t_util(
                            tmp_expr,
                            ASRUtils::type_get_past_pointer(ASRUtils::type_get_past_allocatable(cur_type)),
                            module.get());
                        llvm::Value* class_ptr = tmp;

                        ASR::symbol_t* struct_sym = ASRUtils::symbol_get_past_external(ASRUtils::get_struct_sym_from_struct_expr(tmp_expr));
                        ASR::Struct_t* st = ASR::down_cast<ASR::Struct_t>(struct_sym);
                        llvm::Value* class_hash = llvm::ConstantInt::get(llvm_utils->getIntType(8),
                            llvm::APInt(64, get_class_hash(struct_sym)));

                        // Store back the original hash
                        llvm::Value* t = llvm_utils->create_gep2(class_type, class_ptr, 0);
                        builder->CreateStore(class_hash, t);

                        t = llvm_utils->create_gep2(class_type, class_ptr, 1);
                        llvm_data_type = get_llvm_struct_data_type(st, false);
                        tmp_ = t;
                        tmp = llvm_utils->CreateLoad2(llvm_data_type->getPointerTo(), t);
                    } else {
                        llvm_data_type = llvm_utils->get_type_from_ttype_t_util(tmp_expr, 
                            ASRUtils::type_get_past_array(
                                ASRUtils::type_get_past_pointer(
                                    ASRUtils::type_get_past_allocatable(cur_type))),
                            module.get(), abt);
                    }
                    llvm::Value *cond = builder->CreateICmpNE(
                        builder->CreatePtrToInt(tmp, llvm::Type::getInt64Ty(context)),
                        builder->CreatePtrToInt(
                            llvm::ConstantPointerNull::get(llvm_data_type->getPointerTo()),
                            llvm::Type::getInt64Ty(context)) );
                    llvm_utils->create_if_else(cond, [=]() {
                        llvm::AllocaInst *arg_tmp = llvm_utils->CreateAlloca(*builder, character_type);
                        builder->CreateStore(builder->CreateBitCast(tmp, character_type), arg_tmp);
                        std::vector<llvm::Value*> args = {llvm_utils->CreateLoad(arg_tmp)};
                        builder->CreateCall(free_fn, args);
                        builder->CreateStore(
                            llvm::ConstantPointerNull::get(llvm_data_type->getPointerTo()), tmp_);
                    }, [](){});
                } else {
                    if( LLVM::is_llvm_pointer(*cur_type) ) {
                        tmp = llvm_utils->CreateLoad(tmp);
                    }
#if LLVM_VERSION_MAJOR > 16
                ptr_type[tmp] = llvm_utils->get_type_from_ttype_t_util(tmp_expr,
                        ASRUtils::type_get_past_pointer(
                            ASRUtils::type_get_past_allocatable(cur_type)),
                    module.get(), abt);
#endif
                    llvm::Type* llvm_data_type = llvm_utils->get_type_from_ttype_t_util(tmp_expr,
                        ASRUtils::type_get_past_array(
                            ASRUtils::type_get_past_pointer(
                                ASRUtils::type_get_past_allocatable(cur_type))),
                        module.get(), abt);
                    llvm::Value *cond = arr_descr->get_is_allocated_flag(tmp, llvm_data_type, tmp_expr);
                    llvm_utils->create_if_else(cond, [=]() {
                        call_lfortran_free(free_fn, llvm_data_type);
                    }, [](){});
                }
            }
        }
    }

    void visit_ImplicitDeallocate(const ASR::ImplicitDeallocate_t& x) {
        visit_Deallocate(x);
    }

    void visit_ExplicitDeallocate(const ASR::ExplicitDeallocate_t& x) {
        visit_Deallocate(x);
    }

    void visit_ListConstant(const ASR::ListConstant_t& x) {
        ASR::List_t* list_type = ASR::down_cast<ASR::List_t>(x.m_type);
        bool is_array_type_local = false, is_malloc_array_type_local = false;
        bool is_list_local = false;
        ASR::dimension_t* m_dims_local = nullptr;
        int n_dims_local = -1, a_kind_local = -1;
        llvm::Type* llvm_el_type = llvm_utils->get_type_from_ttype_t(const_cast<ASR::expr_t*>(&x.base), list_type->m_type,
                                    nullptr, ASR::storage_typeType::Default, is_array_type_local,
                                    is_malloc_array_type_local, is_list_local, m_dims_local,
                                    n_dims_local, a_kind_local, module.get());
        std::string type_code = ASRUtils::get_type_code(list_type->m_type);
        int32_t type_size = -1;
        if( ASR::is_a<ASR::String_t>(*list_type->m_type) ||
            LLVM::is_llvm_struct(list_type->m_type) ||
            ASR::is_a<ASR::Complex_t>(*list_type->m_type) ) {
            llvm::DataLayout data_layout(module->getDataLayout());
            type_size = data_layout.getTypeAllocSize(llvm_el_type);
        } else {
            type_size = ASRUtils::extract_kind_from_ttype_t(list_type->m_type);
        }
        llvm::Type* const_list_type = list_api->get_list_type(llvm_el_type, type_code, type_size);
        llvm::Value* const_list = llvm_utils->CreateAlloca(*builder, const_list_type, nullptr, "const_list");
        list_api->list_init(type_code, const_list, module.get(), x.n_args, x.n_args);
        int64_t ptr_loads_copy = ptr_loads;
        for( size_t i = 0; i < x.n_args; i++ ) {
            if (is_argument_of_type_CPtr(x.m_args[i])) {
                ptr_loads = 0;
             } else {
                ptr_loads = 1;
             }
            this->visit_expr(*x.m_args[i]);
            llvm::Value* item = tmp;
            llvm::Value* pos = llvm::ConstantInt::get(context, llvm::APInt(32, i));
            list_api->write_item(const_cast<ASR::expr_t*>(&x.base), const_list, pos, item, list_type->m_type,
                                 false, module.get(), name2memidx);
        }
        ptr_loads = ptr_loads_copy;
        tmp = const_list;
    }

    void visit_DictConstant(const ASR::DictConstant_t& x) {
        llvm::Type* const_dict_type = llvm_utils->get_dict_type(const_cast<ASR::expr_t*>(&x.base), x.m_type, module.get());
        llvm::Value* const_dict = llvm_utils->CreateAlloca(*builder, const_dict_type, nullptr, "const_dict");
        ASR::Dict_t* x_dict = ASR::down_cast<ASR::Dict_t>(x.m_type);
        llvm_utils->set_dict_api(x_dict);
        std::string key_type_code = ASRUtils::get_type_code(x_dict->m_key_type);
        std::string value_type_code = ASRUtils::get_type_code(x_dict->m_value_type);
        llvm_utils->dict_api->dict_init(key_type_code, value_type_code, const_dict, module.get(), x.n_keys);
        int64_t ptr_loads_key = !LLVM::is_llvm_struct(x_dict->m_key_type);
        int64_t ptr_loads_value = !LLVM::is_llvm_struct(x_dict->m_value_type);
        int64_t ptr_loads_copy = ptr_loads;
        for( size_t i = 0; i < x.n_keys; i++ ) {
            ptr_loads = ptr_loads_key;
            visit_expr_wrapper(x.m_keys[i], true);
            llvm::Value* key = tmp;
            ptr_loads = ptr_loads_value;
            visit_expr_wrapper(x.m_values[i], true);
            llvm::Value* value = tmp;
            llvm_utils->dict_api->write_item(const_cast<ASR::expr_t*>(&x.base), const_dict, key, value, module.get(),
                                 x_dict->m_key_type, x_dict->m_value_type, name2memidx);
        }
        ptr_loads = ptr_loads_copy;
        tmp = const_dict;
    }

    void visit_SetConstant(const ASR::SetConstant_t& x) {
        llvm::Type* const_set_type = llvm_utils->get_set_type(const_cast<ASR::expr_t*>(&x.base), x.m_type, module.get());
        llvm::Value* const_set = llvm_utils->CreateAlloca(*builder, const_set_type, nullptr, "const_set");
        ASR::Set_t* x_set = ASR::down_cast<ASR::Set_t>(x.m_type);
        llvm_utils->set_set_api(x_set);
        std::string el_type_code = ASRUtils::get_type_code(x_set->m_type);
        llvm_utils->set_api->set_init(el_type_code, const_set, module.get(), x.n_elements);
        int64_t ptr_loads_el = !LLVM::is_llvm_struct(x_set->m_type);
        int64_t ptr_loads_copy = ptr_loads;
        ptr_loads = ptr_loads_el;
        for( size_t i = 0; i < x.n_elements; i++ ) {
            visit_expr_wrapper(x.m_elements[i], true);
            llvm::Value* element = tmp;
            llvm_utils->set_api->write_item(const_cast<ASR::expr_t*>(&x.base), const_set, element, module.get(),
                                            x_set->m_type, name2memidx);
        }
        ptr_loads = ptr_loads_copy;
        tmp = const_set;
    }

    void visit_TupleConstant(const ASR::TupleConstant_t& x) {
        ASR::Tuple_t* tuple_type = ASR::down_cast<ASR::Tuple_t>(x.m_type);
        std::string type_code = ASRUtils::get_type_code(tuple_type->m_type,
                                                        tuple_type->n_type);
        std::vector<llvm::Type*> llvm_el_types;
        ASR::storage_typeType m_storage = ASR::storage_typeType::Default;
        bool is_array_type = false, is_malloc_array_type = false;
        bool is_list = false;
        ASR::dimension_t* m_dims = nullptr;
        int n_dims = 0, a_kind = -1;
        for( size_t i = 0; i < tuple_type->n_type; i++ ) {
            llvm_el_types.push_back(llvm_utils->get_type_from_ttype_t(const_cast<ASR::expr_t*>(&x.base), tuple_type->m_type[i],
                nullptr, m_storage, is_array_type, is_malloc_array_type,
                is_list, m_dims, n_dims, a_kind, module.get()));
        }
        llvm::Type* const_tuple_type = tuple_api->get_tuple_type(type_code, llvm_el_types);
        llvm::Value* const_tuple = llvm_utils->CreateAlloca(*builder, const_tuple_type, nullptr, "const_tuple");
        std::vector<llvm::Value*> init_values;
        int64_t ptr_loads_copy = ptr_loads;
        for( size_t i = 0; i < x.n_elements; i++ ) {
            if(ASRUtils::is_character(*tuple_type->m_type[i])){
                ptr_loads = 0;
            } else if(!LLVM::is_llvm_struct(tuple_type->m_type[i])) {
                ptr_loads = 2;
            }
            else {
                ptr_loads = ptr_loads_copy;
            }
            this->visit_expr(*x.m_elements[i]);
            init_values.push_back(tmp);
        }
        ptr_loads = ptr_loads_copy;
        tuple_api->tuple_init(const_cast<ASR::expr_t*>(&x.base), const_tuple, init_values, tuple_type,
                              module.get(), name2memidx);
        tmp = const_tuple;
    }

    void visit_IntegerBitLen(const ASR::IntegerBitLen_t& x) {
        if (x.m_value) {
            this->visit_expr_wrapper(x.m_value, true);
            return;
        }
        this->visit_expr(*x.m_a);
        llvm::Value *int_val = tmp;
        int int_kind = ASRUtils::extract_kind_from_ttype_t(x.m_type);
        std::string runtime_func_name = "_lpython_bit_length" + std::to_string(int_kind);
        llvm::Function *fn = module->getFunction(runtime_func_name);
        if (!fn) {
            llvm::FunctionType *function_type = llvm::FunctionType::get(
                    llvm::Type::getInt32Ty(context), {
                        llvm_utils->getIntType(int_kind)
                    }, false);
            fn = llvm::Function::Create(function_type,
                    llvm::Function::ExternalLinkage, runtime_func_name, *module);
        }
        tmp = builder->CreateCall(fn, {int_val});
    }

    void visit_Ichar(const ASR::Ichar_t &x) {
        if (x.m_value) {
            this->visit_expr_wrapper(x.m_value, true);
            return;
        }
        this->visit_expr_wrapper(x.m_arg, true);
        llvm::Value *c = llvm_utils->get_string_data(ASRUtils::get_string_type(x.m_arg), tmp);
        std::string runtime_func_name = "_lfortran_ichar";
        llvm::Function *fn = module->getFunction(runtime_func_name);
        if (!fn) {
            llvm::FunctionType *function_type = llvm::FunctionType::get(
                llvm::Type::getInt32Ty(context), {
                    llvm::Type::getInt8Ty(context)->getPointerTo()
                }, false);
            fn = llvm::Function::Create(function_type,
                    llvm::Function::ExternalLinkage, runtime_func_name, *module);
        }
        tmp = builder->CreateCall(fn, {c});
    }

    void visit_Iachar(const ASR::Iachar_t &x) {
        if (x.m_value) {
            this->visit_expr_wrapper(x.m_value, true);
            return;
        }
        this->visit_expr_load_wrapper(x.m_arg, 0);
        llvm::Value *c = llvm_utils->get_string_data(ASRUtils::get_string_type(x.m_arg), tmp);
        std::string runtime_func_name = "_lfortran_iachar";
        llvm::Function *fn = module->getFunction(runtime_func_name);
        if (!fn) {
            llvm::FunctionType *function_type = llvm::FunctionType::get(
                llvm::Type::getInt32Ty(context), {
                    llvm::Type::getInt8Ty(context)->getPointerTo()
                }, false);
            fn = llvm::Function::Create(function_type,
                    llvm::Function::ExternalLinkage, runtime_func_name, *module);
        }
        tmp = builder->CreateCall(fn, {c});
        if( ASRUtils::extract_kind_from_ttype_t(x.m_type) == 8 ) {
            tmp = builder->CreateSExt(tmp, llvm_utils->getIntType(8));
        }
    }

    void visit_RealSqrt(const ASR::RealSqrt_t &x) {
        if (x.m_value) {
            this->visit_expr_wrapper(x.m_value, true);
            return;
        }
        this->visit_expr(*x.m_arg);
        if (tmp->getType()->isPointerTy()) {
            llvm::Type* llvm_type = llvm_utils->get_type_from_ttype_t_util(x.m_arg, x.m_type, module.get());
            tmp = llvm_utils->CreateLoad2(llvm_type, tmp);
        }
        llvm::Value *c = tmp;
        int64_t kind_value = ASRUtils::extract_kind_from_ttype_t(ASRUtils::expr_type(x.m_arg));
        std::string func_name;
        if (kind_value ==4) {
            func_name = "llvm.sqrt.f32";
        } else {
            func_name = "llvm.sqrt.f64";
        }
        llvm::Type *type = llvm_utils->getFPType(kind_value);
        llvm::Function *fn_sqrt = module->getFunction(func_name);
        if (!fn_sqrt) {
            llvm::FunctionType *function_type = llvm::FunctionType::get(
                    type, {type}, false);
            fn_sqrt = llvm::Function::Create(function_type,
                    llvm::Function::ExternalLinkage, func_name,
                    module.get());
        }
        tmp = builder->CreateCall(fn_sqrt, {c});
    }

    void visit_ListAppend(const ASR::ListAppend_t& x) {
        ASR::List_t* asr_list = ASR::down_cast<ASR::List_t>(ASRUtils::expr_type(x.m_a));
        int64_t ptr_loads_copy = ptr_loads;
        ptr_loads = 0;
        this->visit_expr(*x.m_a);
        llvm::Value* plist = tmp;

        ptr_loads = !(LLVM::is_llvm_struct(asr_list->m_type) || ASRUtils::is_character(*asr_list->m_type)) ? 1 : 0;
        this->visit_expr_wrapper(x.m_ele, true);
        llvm::Value *item = tmp;
        ptr_loads = ptr_loads_copy;

        list_api->append(x.m_a, plist, item, asr_list->m_type, module.get(), name2memidx);
    }

    void visit_UnionInstanceMember(const ASR::UnionInstanceMember_t& x) {
        int64_t ptr_loads_copy = ptr_loads;
        ptr_loads = 0;
        this->visit_expr(*x.m_v);
        ptr_loads = ptr_loads_copy;
        llvm::Value* union_llvm = tmp;
        ASR::Variable_t* member_var = ASR::down_cast<ASR::Variable_t>(
            ASRUtils::symbol_get_past_external(x.m_m));
        ASR::ttype_t* member_type_asr = ASRUtils::get_contained_type(member_var->m_type);
        if( ASR::is_a<ASR::StructType_t>(*member_type_asr) && !ASRUtils::is_class_type(member_type_asr) ) {
            current_der_type_name = ASRUtils::symbol_name(member_var->m_type_declaration);
        }
        member_type_asr = member_var->m_type;
        llvm::Type* member_type_llvm = llvm_utils->getMemberType(member_type_asr, member_var, module.get())->getPointerTo();
        tmp = builder->CreateBitCast(union_llvm, member_type_llvm);
        if( is_assignment_target ) {
            return ;
        }
        if( ptr_loads > 0 ) {
            tmp = llvm_utils->CreateLoad(tmp);
        }
    }

    void visit_ListItem(const ASR::ListItem_t& x) {
        ASR::ttype_t* el_type = ASRUtils::get_contained_type(
                                        ASRUtils::expr_type(x.m_a));
        visit_expr_load_wrapper(x.m_a, 0);
        llvm::Value* plist = tmp;

        visit_expr_load_wrapper(x.m_pos, 1, true);
        llvm::Value *pos = tmp;

        tmp = list_api->read_item_using_ttype(el_type, plist, pos, compiler_options.enable_bounds_checking, module.get(),
                (LLVM::is_llvm_struct(el_type) || ptr_loads == 0));
    }

    void visit_DictItem(const ASR::DictItem_t& x) {
        ASR::Dict_t* dict_type = ASR::down_cast<ASR::Dict_t>(
                                    ASRUtils::expr_type(x.m_a));
        int64_t ptr_loads_copy = ptr_loads;
        ptr_loads = 0;
        this->visit_expr(*x.m_a);
        llvm::Value* pdict = tmp;

        ptr_loads = !(LLVM::is_llvm_struct(dict_type->m_key_type) || ASRUtils::is_character(*dict_type->m_key_type)) ? 1 : 0;
        this->visit_expr_wrapper(x.m_key, true);
        ptr_loads = ptr_loads_copy;
        llvm::Value *key = tmp;
        if (x.m_default) {
            llvm::Type *val_type = llvm_utils->get_type_from_ttype_t_util(x.m_a, dict_type->m_value_type, module.get());
            llvm::Value *def_value_ptr = llvm_utils->CreateAlloca(*builder, val_type);
            ptr_loads = !LLVM::is_llvm_struct(dict_type->m_value_type);
            this->visit_expr_wrapper(x.m_default, true);
            ptr_loads = ptr_loads_copy;
            builder->CreateStore(tmp, def_value_ptr);
            llvm_utils->set_dict_api(dict_type);
            tmp = llvm_utils->dict_api->get_item(x.m_a, pdict, key, module.get(), dict_type, def_value_ptr,
                                  LLVM::is_llvm_struct(dict_type->m_value_type));
        } else {
            llvm_utils->set_dict_api(dict_type);
            tmp = llvm_utils->dict_api->read_item(x.m_a, pdict, key, module.get(), dict_type,
                                    compiler_options.enable_bounds_checking,
                                    LLVM::is_llvm_struct(dict_type->m_value_type));
        }
    }

    void visit_DictPop(const ASR::DictPop_t& x) {
        ASR::Dict_t* dict_type = ASR::down_cast<ASR::Dict_t>(
                                    ASRUtils::expr_type(x.m_a));
        int64_t ptr_loads_copy = ptr_loads;
        ptr_loads = 0;
        this->visit_expr(*x.m_a);
        llvm::Value* pdict = tmp;

        ptr_loads = !(LLVM::is_llvm_struct(dict_type->m_key_type) || ASRUtils::is_character(*dict_type->m_key_type)) ? 1 : 0;
        this->visit_expr_wrapper(x.m_key, true);
        ptr_loads = ptr_loads_copy;
        llvm::Value *key = tmp;

        llvm_utils->set_dict_api(dict_type);
        tmp = llvm_utils->dict_api->pop_item(x.m_a, pdict, key, module.get(), dict_type,
                                 LLVM::is_llvm_struct(dict_type->m_value_type));
    }

    void visit_ListLen(const ASR::ListLen_t& x) {
        if (x.m_value) {
            this->visit_expr(*x.m_value);
        } else {
            int64_t ptr_loads_copy = ptr_loads;
            ptr_loads = 0;
            this->visit_expr(*x.m_arg);
            ptr_loads = ptr_loads_copy;
            llvm::Value* plist = tmp;

            std::string type_code = ASRUtils::get_type_code(
                ASRUtils::get_contained_type(ASRUtils::expr_type(x.m_arg)));
            llvm::Type* list_type = list_api->get_list_type(nullptr, type_code, 0);
            tmp = list_api->len_using_type(list_type, plist);
        }
    }

    void visit_ListCompare(const ASR::ListCompare_t x) {
        int64_t ptr_loads_copy = ptr_loads;
        ptr_loads = 0;
        this->visit_expr(*x.m_left);
        llvm::Value* left = tmp;
        this->visit_expr(*x.m_right);
        llvm::Value* right = tmp;
        ptr_loads = ptr_loads_copy;

        ASR::ttype_t* int32_type = ASRUtils::TYPE(ASR::make_Integer_t(al, x.base.base.loc, 4));

        if(x.m_op == ASR::cmpopType::Eq || x.m_op == ASR::cmpopType::NotEq) {
            tmp = llvm_utils->is_equal_by_value(left, right, module.get(),
                    ASRUtils::expr_type(x.m_left));
            if (x.m_op == ASR::cmpopType::NotEq) {
                tmp = builder->CreateNot(tmp);
            }
        }
        else if(x.m_op == ASR::cmpopType::Lt) {
            tmp = llvm_utils->is_ineq_by_value(left, right, module.get(),
                    ASRUtils::expr_type(x.m_left), 0, int32_type);
        }
        else if(x.m_op == ASR::cmpopType::LtE) {
            tmp = llvm_utils->is_ineq_by_value(left, right, module.get(),
                    ASRUtils::expr_type(x.m_left), 1, int32_type);
        }
        else if(x.m_op == ASR::cmpopType::Gt) {
            tmp = llvm_utils->is_ineq_by_value(left, right, module.get(),
                    ASRUtils::expr_type(x.m_left), 2, int32_type);
        }
        else if(x.m_op == ASR::cmpopType::GtE) {
            tmp = llvm_utils->is_ineq_by_value(left, right, module.get(),
                    ASRUtils::expr_type(x.m_left), 3, int32_type);
        }
    }

    void visit_DictLen(const ASR::DictLen_t& x) {
        if (x.m_value) {
            this->visit_expr(*x.m_value);
            return ;
        }

        int64_t ptr_loads_copy = ptr_loads;
        ptr_loads = 0;
        this->visit_expr(*x.m_arg);
        ptr_loads = ptr_loads_copy;
        llvm::Value* pdict = tmp;
        ASR::Dict_t* x_dict = ASR::down_cast<ASR::Dict_t>(ASRUtils::expr_type(x.m_arg));
        llvm_utils->set_dict_api(x_dict);
        tmp = llvm_utils->dict_api->len(pdict);
    }

    void visit_SetLen(const ASR::SetLen_t& x) {
        if (x.m_value) {
            this->visit_expr(*x.m_value);
            return ;
        }

        int64_t ptr_loads_copy = ptr_loads;
        ptr_loads = 0;
        this->visit_expr(*x.m_arg);
        ptr_loads = ptr_loads_copy;
        llvm::Value* pset = tmp;
        ASR::Set_t* x_set = ASR::down_cast<ASR::Set_t>(ASRUtils::expr_type(x.m_arg));
        llvm_utils->set_set_api(x_set);
        tmp = llvm_utils->set_api->len(pset);
    }

    void visit_ListInsert(const ASR::ListInsert_t& x) {
        ASR::List_t* asr_list = ASR::down_cast<ASR::List_t>(
                                    ASRUtils::expr_type(x.m_a));

        this->visit_expr_load_wrapper(x.m_a, 0);
        llvm::Value* plist = tmp;

        this->visit_expr_load_wrapper(x.m_pos, 1, true);
        llvm::Value *pos = tmp;

        visit_expr_load_wrapper(x.m_ele,
            !(LLVM::is_llvm_struct(asr_list->m_type) || ASRUtils::is_character(*asr_list->m_type)) ? 1 : 0,
            true);
        llvm::Value *item = tmp;

        list_api->insert_item(x.m_a, plist, pos, item, asr_list->m_type, module.get(), name2memidx);
    }

    void visit_DictInsert(const ASR::DictInsert_t& x) {
        ASR::Dict_t* dict_type = ASR::down_cast<ASR::Dict_t>(
                                    ASRUtils::expr_type(x.m_a));
        int64_t ptr_loads_copy = ptr_loads;
        ptr_loads = 0;
        this->visit_expr(*x.m_a);
        llvm::Value* pdict = tmp;

        ptr_loads = !LLVM::is_llvm_struct(dict_type->m_key_type);
        this->visit_expr_wrapper(x.m_key, true);
        llvm::Value *key = tmp;
        visit_expr_load_wrapper(x.m_value,
            !(LLVM::is_llvm_struct(dict_type->m_value_type) || ASRUtils::is_character(*dict_type->m_value_type)) ? 1 : 0,
            true);
        llvm::Value *value = tmp;
        ptr_loads = ptr_loads_copy;

        llvm_utils->set_dict_api(dict_type);
        llvm_utils->dict_api->write_item(x.m_a, pdict, key, value, module.get(),
                             dict_type->m_key_type,
                             dict_type->m_value_type, name2memidx);
    }

    void visit_Expr(const ASR::Expr_t& x) {
        this->visit_expr_wrapper(x.m_expression, false);
    }

    void visit_ListRemove(const ASR::ListRemove_t& x) {
        ASR::ttype_t* asr_el_type = ASRUtils::get_contained_type(ASRUtils::expr_type(x.m_a));
        int64_t ptr_loads_copy = ptr_loads;
        ptr_loads = 0;
        this->visit_expr(*x.m_a);
        llvm::Value* plist = tmp;

        ptr_loads = !LLVM::is_llvm_struct(asr_el_type);
        this->visit_expr_wrapper(x.m_ele, true);
        ptr_loads = ptr_loads_copy;
        llvm::Value *item = tmp;
        list_api->remove(plist, item, asr_el_type, module.get());
    }

    void visit_ListCount(const ASR::ListCount_t& x) {
        ASR::ttype_t *asr_el_type = ASRUtils::get_contained_type(ASRUtils::expr_type(x.m_arg));
        int64_t ptr_loads_copy = ptr_loads;
        ptr_loads = 0;
        this->visit_expr(*x.m_arg);
        llvm::Value* plist = tmp;

        ptr_loads = !LLVM::is_llvm_struct(asr_el_type);
        this->visit_expr_wrapper(x.m_ele, true);
        ptr_loads = ptr_loads_copy;
        llvm::Value *item = tmp;
        tmp = list_api->count(plist, item, asr_el_type, module.get());
    }

    void generate_ListIndex(ASR::expr_t* m_arg, ASR::expr_t* m_ele,
            ASR::expr_t* m_start=nullptr, ASR::expr_t* m_end=nullptr) {
        ASR::ttype_t *asr_el_type = ASRUtils::get_contained_type(ASRUtils::expr_type(m_arg));
        int64_t ptr_loads_copy = ptr_loads;
        ptr_loads = 0;
        this->visit_expr(*m_arg);
        llvm::Value* plist = tmp;

        ptr_loads = !LLVM::is_llvm_struct(asr_el_type);
        this->visit_expr_wrapper(m_ele, true);
        llvm::Value *item = tmp;

        llvm::Value* start = nullptr;
        llvm::Value* end = nullptr;
        if(m_start) {
            ptr_loads = 2;
            this->visit_expr_wrapper(m_start, true);
            start = tmp;
        }
        if(m_end) {
            ptr_loads = 2;
            this->visit_expr_wrapper(m_end, true);
            end = tmp;
        }

        ptr_loads = ptr_loads_copy;
        tmp = list_api->index(plist, item, start, end, asr_el_type, module.get());
    }

    void generate_Exp(ASR::expr_t* m_arg) {
        this->visit_expr_wrapper(m_arg, true);
        llvm::Value *item = tmp;
        tmp = builder->CreateUnaryIntrinsic(llvm::Intrinsic::exp, item);
    }

    void generate_Exp2(ASR::expr_t* m_arg) {
        this->visit_expr_wrapper(m_arg, true);
        llvm::Value *item = tmp;
        tmp = builder->CreateUnaryIntrinsic(llvm::Intrinsic::exp2, item);
    }

    void generate_Expm1(ASR::expr_t* m_arg) {
        this->visit_expr_wrapper(m_arg, true);
        llvm::Value *item = tmp;
        llvm::Value* exp = builder->CreateUnaryIntrinsic(llvm::Intrinsic::exp, item);
        llvm::Value* one = llvm::ConstantFP::get(builder->getFloatTy(), 1.0);
        tmp = builder->CreateFSub(exp, one);
    }

    void generate_ListReverse(ASR::expr_t* m_arg) {
        ASR::ttype_t* asr_el_type = ASRUtils::get_contained_type(ASRUtils::expr_type(m_arg));
        int64_t ptr_loads_copy = ptr_loads;
        ptr_loads = 0;
        this->visit_expr(*m_arg);
        llvm::Value* plist = tmp;

        ptr_loads = !LLVM::is_llvm_struct(asr_el_type);
        ptr_loads = ptr_loads_copy;

        list_api->reverse(asr_el_type, plist, module.get());
    }

    void generate_ListPop_0(ASR::expr_t* m_arg) {
        ASR::ttype_t* asr_el_type = ASRUtils::get_contained_type(ASRUtils::expr_type(m_arg));
        int64_t ptr_loads_copy = ptr_loads;
        ptr_loads = 0;
        this->visit_expr(*m_arg);
        llvm::Value* plist = tmp;

        ptr_loads = !LLVM::is_llvm_struct(asr_el_type);
        ptr_loads = ptr_loads_copy;
        tmp = list_api->pop_last(plist, asr_el_type, module.get());
    }

    void generate_ListPop_1(ASR::expr_t* m_arg, ASR::expr_t* m_ele) {
        ASR::ttype_t* asr_el_type = ASRUtils::get_contained_type(ASRUtils::expr_type(m_arg));
        int64_t ptr_loads_copy = ptr_loads;
        ptr_loads = 0;
        this->visit_expr(*m_arg);
        llvm::Value* plist = tmp;

        ptr_loads = 2;
        this->visit_expr_wrapper(m_ele, true);
        ptr_loads = ptr_loads_copy;
        llvm::Value *pos = tmp;
        tmp = list_api->pop_position(m_arg, plist, pos, asr_el_type, module.get(), name2memidx);
    }

    void generate_ListReserve(ASR::expr_t* m_arg, ASR::expr_t* m_ele) {
        // For now, this only handles lists
        ASR::ttype_t* asr_el_type = ASRUtils::get_contained_type(ASRUtils::expr_type(m_arg));
        int64_t ptr_loads_copy = ptr_loads;
        ptr_loads = 0;
        this->visit_expr(*m_arg);
        llvm::Value* plist = tmp;

        ptr_loads = 2;
        this->visit_expr_wrapper(m_ele, true);
        ptr_loads = ptr_loads_copy;
        llvm::Value* n = tmp;
        list_api->reserve(plist, n, asr_el_type, module.get());
    }

    void generate_DictElems(ASR::expr_t* m_arg, bool key_or_value) {
        ASR::Dict_t* dict_type = ASR::down_cast<ASR::Dict_t>(
                                    ASRUtils::expr_type(m_arg));
        ASR::ttype_t* el_type = key_or_value == 0 ?
                                    dict_type->m_key_type : dict_type->m_value_type;

        int64_t ptr_loads_copy = ptr_loads;
        ptr_loads = 0;
        this->visit_expr(*m_arg);
        llvm::Value* pdict = tmp;

        ptr_loads = ptr_loads_copy;

        bool is_array_type_local = false, is_malloc_array_type_local = false;
        bool is_list_local = false;
        ASR::dimension_t* m_dims_local = nullptr;
        int n_dims_local = -1, a_kind_local = -1;
        llvm::Type* llvm_el_type = llvm_utils->get_type_from_ttype_t(m_arg, el_type, nullptr,
                                    ASR::storage_typeType::Default, is_array_type_local,
                                    is_malloc_array_type_local, is_list_local, m_dims_local,
                                    n_dims_local, a_kind_local, module.get());
        std::string type_code = ASRUtils::get_type_code(el_type);
        int32_t type_size = -1;
        if( ASR::is_a<ASR::String_t>(*el_type) ||
            LLVM::is_llvm_struct(el_type) ||
            ASR::is_a<ASR::Complex_t>(*el_type) ) {
            llvm::DataLayout data_layout(module->getDataLayout());
            type_size = data_layout.getTypeAllocSize(llvm_el_type);
        } else {
            type_size = ASRUtils::extract_kind_from_ttype_t(el_type);
        }
        llvm::Type* el_list_type = list_api->get_list_type(llvm_el_type, type_code, type_size);
        llvm::Value* el_list = llvm_utils->CreateAlloca(*builder, el_list_type, nullptr, key_or_value == 0 ?
                                                    "keys_list" : "values_list");
        list_api->list_init(type_code, el_list, module.get(), 0, 0);

        llvm_utils->set_dict_api(dict_type);
        llvm_utils->dict_api->get_elements_list(m_arg, pdict, el_list, dict_type->m_key_type,
                                                dict_type->m_value_type, module.get(),
                                                name2memidx, key_or_value);
        tmp = el_list;
    }

    void generate_SetAdd(ASR::expr_t* m_arg, ASR::expr_t* m_ele) {
        ASR::Set_t* set_type = ASR::down_cast<ASR::Set_t>(
                                    ASRUtils::expr_type(m_arg));
        ASR::ttype_t* asr_el_type = ASRUtils::get_contained_type(ASRUtils::expr_type(m_arg));
        int64_t ptr_loads_copy = ptr_loads;
        ptr_loads = 0;
        this->visit_expr(*m_arg);
        llvm::Value* pset = tmp;

        ptr_loads = 2;
        this->visit_expr_wrapper(m_ele, true);
        ptr_loads = ptr_loads_copy;
        llvm::Value *el = tmp;
        llvm_utils->set_set_api(set_type);
        llvm_utils->set_api->write_item(m_arg, pset, el, module.get(), asr_el_type, name2memidx);
    }

    void generate_SetRemove(ASR::expr_t* m_arg, ASR::expr_t* m_ele) {
        ASR::Set_t* set_type = ASR::down_cast<ASR::Set_t>(
                                    ASRUtils::expr_type(m_arg));
        ASR::ttype_t* asr_el_type = ASRUtils::get_contained_type(ASRUtils::expr_type(m_arg));
        int64_t ptr_loads_copy = ptr_loads;
        ptr_loads = 0;
        this->visit_expr(*m_arg);
        llvm::Value* pset = tmp;

        ptr_loads = 2;
        this->visit_expr_wrapper(m_ele, true);
        ptr_loads = ptr_loads_copy;
        llvm::Value *el = tmp;
        llvm_utils->set_set_api(set_type);
        llvm_utils->set_api->remove_item(pset, el, module.get(), asr_el_type);
    }

    void visit_IntrinsicElementalFunction(const ASR::IntrinsicElementalFunction_t& x) {
        if (x.m_value) {
            this->visit_expr_wrapper(x.m_value, true);
            return;
        }
        switch (static_cast<ASRUtils::IntrinsicElementalFunctions>(x.m_intrinsic_id)) {
            case ASRUtils::IntrinsicElementalFunctions::ListIndex: {
                ASR::expr_t* m_arg = x.m_args[0];
                ASR::expr_t* m_ele = x.m_args[1];
                ASR::expr_t* m_start = nullptr;
                ASR::expr_t* m_end = nullptr;
                switch (x.m_overload_id) {
                    case 0: {
                        break ;
                    }
                    case 1: {
                        m_start = x.m_args[2];
                        break ;
                    }
                    case 2: {
                        m_start = x.m_args[2];
                        m_end = x.m_args[3];
                        break ;
                    }
                    default: {
                        throw CodeGenError("list.index accepts at most four arguments",
                                            x.base.base.loc);
                    }
                }
                generate_ListIndex(m_arg, m_ele, m_start, m_end);
                break ;
            }
            case ASRUtils::IntrinsicElementalFunctions::ListReverse: {
                generate_ListReverse(x.m_args[0]);
                break;
            }
            case ASRUtils::IntrinsicElementalFunctions::ListPop: {
                switch(x.m_overload_id) {
                    case 0:
                        generate_ListPop_0(x.m_args[0]);
                        break;
                    case 1:
                        generate_ListPop_1(x.m_args[0], x.m_args[1]);
                        break;
                }
                break;
            }
            case ASRUtils::IntrinsicElementalFunctions::ListReserve: {
                generate_ListReserve(x.m_args[0], x.m_args[1]);
                break;
            }
            case ASRUtils::IntrinsicElementalFunctions::DictKeys: {
                generate_DictElems(x.m_args[0], 0);
                break;
            }
            case ASRUtils::IntrinsicElementalFunctions::DictValues: {
                generate_DictElems(x.m_args[0], 1);
                break;
            }
            case ASRUtils::IntrinsicElementalFunctions::SetAdd: {
                generate_SetAdd(x.m_args[0], x.m_args[1]);
                break;
            }
            case ASRUtils::IntrinsicElementalFunctions::SetRemove: {
                generate_SetRemove(x.m_args[0], x.m_args[1]);
                break;
            }
            case ASRUtils::IntrinsicElementalFunctions::Exp: {
                switch (x.m_overload_id) {
                    case 0: {
                        ASR::expr_t* m_arg = x.m_args[0];
                        generate_Exp(m_arg);
                        break ;
                    }
                    default: {
                        throw CodeGenError("exp() only accepts one argument",
                                            x.base.base.loc);
                    }
                }
                break ;
            }
            case ASRUtils::IntrinsicElementalFunctions::Exp2: {
                switch (x.m_overload_id) {
                    case 0: {
                        ASR::expr_t* m_arg = x.m_args[0];
                        generate_Exp2(m_arg);
                        break ;
                    }
                    default: {
                        throw CodeGenError("exp2() only accepts one argument",
                                            x.base.base.loc);
                    }
                }
                break ;
            }
            case ASRUtils::IntrinsicElementalFunctions::CommandArgumentCount: {
                break;
            }
            case ASRUtils::IntrinsicElementalFunctions::Expm1: {
                switch (x.m_overload_id) {
                    case 0: {
                        ASR::expr_t* m_arg = x.m_args[0];
                        generate_Expm1(m_arg);
                        break ;
                    }
                    default: {
                        throw CodeGenError("expm1() only accepts one argument",
                                            x.base.base.loc);
                    }
                }
                break ;
            }
            case ASRUtils::IntrinsicElementalFunctions::FlipSign: {
                Vec<ASR::call_arg_t> args;
                args.reserve(al, 2);
                ASR::call_arg_t arg0_, arg1_;
                arg0_.loc = x.m_args[0]->base.loc, arg0_.m_value = x.m_args[0];
                args.push_back(al, arg0_);
                arg1_.loc = x.m_args[1]->base.loc, arg1_.m_value = x.m_args[1];
                args.push_back(al, arg1_);
                generate_flip_sign(args.p);
                break;
            }
            case ASRUtils::IntrinsicElementalFunctions::FMA: {
                Vec<ASR::call_arg_t> args;
                args.reserve(al, 3);
                ASR::call_arg_t arg0_, arg1_, arg2_;
                arg0_.loc = x.m_args[0]->base.loc, arg0_.m_value = x.m_args[0];
                args.push_back(al, arg0_);
                arg1_.loc = x.m_args[1]->base.loc, arg1_.m_value = x.m_args[1];
                args.push_back(al, arg1_);
                arg2_.loc = x.m_args[2]->base.loc, arg2_.m_value = x.m_args[2];
                args.push_back(al, arg2_);
                generate_fma(args.p);
                break;
            }
            case ASRUtils::IntrinsicElementalFunctions::SignFromValue: {
                Vec<ASR::call_arg_t> args;
                args.reserve(al, 2);
                ASR::call_arg_t arg0_, arg1_;
                arg0_.loc = x.m_args[0]->base.loc, arg0_.m_value = x.m_args[0];
                args.push_back(al, arg0_);
                arg1_.loc = x.m_args[1]->base.loc, arg1_.m_value = x.m_args[1];
                args.push_back(al, arg1_);
                generate_sign_from_value(args.p);
                break;
            }
            default: {
                throw CodeGenError("Either the '" + ASRUtils::IntrinsicElementalFunctionRegistry::
                        get_intrinsic_function_name(x.m_intrinsic_id) +
                        "' intrinsic is not implemented by LLVM backend or "
                        "the compile-time value is not available", x.base.base.loc);
            }
        }
    }

    void visit_IntrinsicImpureFunction(const ASR::IntrinsicImpureFunction_t &x) {
        switch (static_cast<ASRUtils::IntrinsicImpureFunctions>(x.m_impure_intrinsic_id)) {
            case ASRUtils::IntrinsicImpureFunctions::IsIostatEnd : {
                this->visit_expr(*x.m_args[0]);
                tmp = builder->CreateICmpEQ(tmp, llvm::ConstantInt::get(context, llvm::APInt(32, -1, true)));
                break ;
            } case ASRUtils::IntrinsicImpureFunctions::IsIostatEor : {
                this->visit_expr(*x.m_args[0]);
                tmp = builder->CreateICmpEQ(tmp, llvm::ConstantInt::get(context, llvm::APInt(32, -2, true)));
                break ;
            } case ASRUtils::IntrinsicImpureFunctions::Allocated : {
                handle_allocated(x.m_args[0]);
                break ;
            } default: {
                throw CodeGenError( ASRUtils::get_impure_intrinsic_name(x.m_impure_intrinsic_id) +
                        " is not implemented by LLVM backend.", x.base.base.loc);
            }
        }
    }

    void visit_TypeInquiry(const ASR::TypeInquiry_t &x) {
        this->visit_expr(*x.m_value);
    }

    void visit_ListClear(const ASR::ListClear_t& x) {
        int64_t ptr_loads_copy = ptr_loads;
        ptr_loads = 0;
        this->visit_expr(*x.m_a);
        llvm::Value* plist = tmp;
        ptr_loads = ptr_loads_copy;
        llvm::Type* list_type = llvm_utils->get_type_from_ttype_t_util(x.m_a, ASRUtils::expr_type(x.m_a), module.get());
        list_api->list_clear_using_type(list_type, plist);
    }

    void visit_ListRepeat(const ASR::ListRepeat_t& x) {
        this->visit_expr_wrapper(x.m_left, true);
        llvm::Value *left = tmp;
        ptr_loads = 2;      // right is int always
        this->visit_expr_wrapper(x.m_right, true);
        llvm::Value *right = tmp;

        ASR::List_t* list_type = ASR::down_cast<ASR::List_t>(x.m_type);
        bool is_array_type_local = false, is_malloc_array_type_local = false;
        bool is_list_local = false;
        ASR::dimension_t* m_dims_local = nullptr;
        int n_dims_local = -1, a_kind_local = -1;
        llvm::Type* llvm_el_type = llvm_utils->get_type_from_ttype_t(const_cast<ASR::expr_t*>(&x.base), list_type->m_type,
            nullptr, ASR::storage_typeType::Default, is_array_type_local,
            is_malloc_array_type_local, is_list_local, m_dims_local,
            n_dims_local, a_kind_local, module.get());
        std::string type_code = ASRUtils::get_type_code(list_type->m_type);
        int32_t type_size = -1;
        if( ASR::is_a<ASR::String_t>(*list_type->m_type) ||
            LLVM::is_llvm_struct(list_type->m_type) ||
            ASR::is_a<ASR::Complex_t>(*list_type->m_type) ) {
            llvm::DataLayout data_layout(module->getDataLayout());
            type_size = data_layout.getTypeAllocSize(llvm_el_type);
        } else {
            type_size = ASRUtils::extract_kind_from_ttype_t(list_type->m_type);
        }
        llvm::Type* repeat_list_type = list_api->get_list_type(llvm_el_type, type_code, type_size);
        llvm::Value* repeat_list = llvm_utils->CreateAlloca(*builder, repeat_list_type, nullptr, "repeat_list");
        llvm::Value* left_len = list_api->len_using_type(repeat_list_type, left);
        llvm::Value* capacity = builder->CreateMul(left_len, right);
        list_api->list_init(type_code, repeat_list, module.get(), capacity, capacity);
        int64_t ptr_loads_copy = ptr_loads;
        ptr_loads = 1;
        list_api->list_repeat_copy(list_type, repeat_list, left, right, left_len, module.get());
        ptr_loads = ptr_loads_copy;
        tmp = repeat_list;
    }

    void visit_TupleCompare(const ASR::TupleCompare_t& x) {
        int64_t ptr_loads_copy = ptr_loads;
        ptr_loads = 0;
        this->visit_expr(*x.m_left);
        llvm::Value* left = tmp;
        this->visit_expr(*x.m_right);
        llvm::Value* right = tmp;
        ptr_loads = ptr_loads_copy;
        if(x.m_op == ASR::cmpopType::Eq || x.m_op == ASR::cmpopType::NotEq) {
            tmp = llvm_utils->is_equal_by_value(left, right, module.get(),
                    ASRUtils::expr_type(x.m_left));
            if (x.m_op == ASR::cmpopType::NotEq) {
                tmp = builder->CreateNot(tmp);
            }
        }
        else if(x.m_op == ASR::cmpopType::Lt) {
            tmp = llvm_utils->is_ineq_by_value(left, right, module.get(),
                    ASRUtils::expr_type(x.m_left), 0);
        }
        else if(x.m_op == ASR::cmpopType::LtE) {
            tmp = llvm_utils->is_ineq_by_value(left, right, module.get(),
                    ASRUtils::expr_type(x.m_left), 1);
        }
        else if(x.m_op == ASR::cmpopType::Gt) {
            tmp = llvm_utils->is_ineq_by_value(left, right, module.get(),
                    ASRUtils::expr_type(x.m_left), 2);
        }
        else if(x.m_op == ASR::cmpopType::GtE) {
            tmp = llvm_utils->is_ineq_by_value(left, right, module.get(),
                    ASRUtils::expr_type(x.m_left), 3);
        }
    }

    void visit_TupleLen(const ASR::TupleLen_t& x) {
        LCOMPILERS_ASSERT(x.m_value);
        this->visit_expr(*x.m_value);
    }

    void visit_TupleItem(const ASR::TupleItem_t& x) {
        int64_t ptr_loads_copy = ptr_loads;
        ptr_loads = 0;
        this->visit_expr(*x.m_a);
        ptr_loads = ptr_loads_copy;
        llvm::Value* ptuple = tmp;

        this->visit_expr_wrapper(x.m_pos, true);
        llvm::Value *pos = tmp;

        llvm::Type* el_type = llvm_utils->get_type_from_ttype_t_util(x.m_a, x.m_type, llvm_utils->module); 
        tmp = tuple_api->read_item_using_pos_value(el_type, ptuple, ASR::down_cast<ASR::Tuple_t>(ASRUtils::expr_type(x.m_a)), pos, LLVM::is_llvm_struct(x.m_type));
    }

    void visit_TupleConcat(const ASR::TupleConcat_t& x) {
        int64_t ptr_loads_copy = ptr_loads;
        ptr_loads = 0;
        this->visit_expr(*x.m_left);
        llvm::Value* left = tmp;
        this->visit_expr(*x.m_right);
        llvm::Value* right = tmp;
        ptr_loads = ptr_loads_copy;

        ASR::Tuple_t* tuple_type_left = ASR::down_cast<ASR::Tuple_t>(ASRUtils::expr_type(x.m_left));
        std::string type_code_left = ASRUtils::get_type_code(tuple_type_left->m_type,
                                                        tuple_type_left->n_type);
        ASR::Tuple_t* tuple_type_right = ASR::down_cast<ASR::Tuple_t>(ASRUtils::expr_type(x.m_right));
        std::string type_code_right = ASRUtils::get_type_code(tuple_type_right->m_type,
                                                        tuple_type_right->n_type);
        Vec<ASR::ttype_t*> v_type;
        v_type.reserve(al, tuple_type_left->n_type + tuple_type_right->n_type);
        std::string type_code = type_code_left + type_code_right;
        std::vector<llvm::Type*> llvm_el_types;
        ASR::storage_typeType m_storage = ASR::storage_typeType::Default;
        bool is_array_type = false, is_malloc_array_type = false;
        bool is_list = false;
        ASR::dimension_t* m_dims = nullptr;
        int n_dims = 0, a_kind = -1;
        for( size_t i = 0; i < tuple_type_left->n_type; i++ ) {
            llvm_el_types.push_back(llvm_utils->get_type_from_ttype_t(x.m_left, tuple_type_left->m_type[i],
                nullptr, m_storage, is_array_type, is_malloc_array_type,
                is_list, m_dims, n_dims, a_kind, module.get()));
            v_type.push_back(al, tuple_type_left->m_type[i]);
        }
        is_array_type = false; is_malloc_array_type = false;
        is_list = false;
        m_dims = nullptr;
        n_dims = 0; a_kind = -1;
        for( size_t i = 0; i < tuple_type_right->n_type; i++ ) {
            llvm_el_types.push_back(llvm_utils->get_type_from_ttype_t(x.m_right, tuple_type_right->m_type[i],
                nullptr, m_storage, is_array_type, is_malloc_array_type,
                is_list, m_dims, n_dims, a_kind, module.get()));
            v_type.push_back(al, tuple_type_right->m_type[i]);
        }
        llvm::Type* concat_tuple_type = tuple_api->get_tuple_type(type_code, llvm_el_types);
        llvm::Value* concat_tuple = llvm_utils->CreateAlloca(*builder, concat_tuple_type, nullptr, "concat_tuple");
        ASR::Tuple_t* tuple_type = (ASR::Tuple_t*)(ASR::make_Tuple_t(
                                    al, x.base.base.loc, v_type.p, v_type.n));
        tuple_api->concat(x.m_left, left, right, tuple_type_left, tuple_type_right, concat_tuple,
                          tuple_type, module.get(), name2memidx);
        tmp = concat_tuple;
    }

    void visit_ArrayItem(const ASR::ArrayItem_t& x) {
        if (x.m_value) {
            this->visit_expr_wrapper(x.m_value, true);
            return;
        }
        ASR::ttype_t* x_mv_type = ASRUtils::expr_type(x.m_v);
        llvm::Value* array = nullptr;
        ASR::Variable_t *v = nullptr;
        if( ASR::is_a<ASR::Var_t>(*x.m_v) ) {
            v = ASRUtils::EXPR2VAR(x.m_v);
            uint32_t v_h = get_hash((ASR::asr_t*)v);
            array = llvm_symtab[v_h];
            if (ASR::is_a<ASR::StructType_t>(*ASRUtils::extract_type(v->m_type))) {
                ASR::Struct_t* der_symbol = ASR::down_cast<ASR::Struct_t>(
                    ASRUtils::symbol_get_past_external(v->m_type_declaration));
                current_der_type_name = std::string(der_symbol->m_name);
            }
        } else {
            int64_t ptr_loads_copy = ptr_loads;
            ptr_loads = 0;
            this->visit_expr(*x.m_v);
            ptr_loads = ptr_loads_copy;
            array = tmp;
        }

        ASR::dimension_t* m_dims;
        int n_dims = ASRUtils::extract_dimensions_from_ttype(x_mv_type, m_dims);
        if (ASRUtils::is_character(*x.m_type) && n_dims == 0) {
            // String indexing:
            if (x.n_args != 1) {
                throw CodeGenError("Only string(a) supported for now.", x.base.base.loc);
            }
            LCOMPILERS_ASSERT(ASR::is_a<ASR::Var_t>(*x.m_args[0].m_right));
            this->visit_expr_wrapper(x.m_args[0].m_right, true);
            llvm::Value *p = nullptr;
            llvm::Value *idx = tmp;
            llvm::Value *str = llvm_utils->CreateLoad(array);
            if( is_assignment_target ) {
                idx = builder->CreateSub(idx, llvm::ConstantInt::get(context, llvm::APInt(32, 1)));
                std::vector<llvm::Value*> idx_vec = {idx};
                p = llvm_utils->CreateGEP(str, idx_vec);
            } else {
                ASR::String_t* str_type = ASR::down_cast<ASR::String_t>(ASRUtils::extract_type(expr_type(x.m_v)));
                llvm::Value* str_len = llvm_utils->get_string_length(str_type, str);
                p = lfortran_str_item(str, str_len, idx);
            }
            // TODO: Currently the string starts at the right location, but goes to the end of the original string.
            // We have to allocate a new string, copy it and add null termination.

            tmp = p;
            if( ptr_loads == 0 ) {
                tmp = llvm_utils->CreateAlloca(*builder, character_type);
                builder->CreateStore(p, tmp);
            }

            //tmp = p;
        } else {
            // Array indexing:
            std::vector<llvm::Value*> indices;
            for( size_t r = 0; r < x.n_args; r++ ) {
                ASR::array_index_t curr_idx = x.m_args[r];
                int64_t ptr_loads_copy = ptr_loads;
                ptr_loads = 2;
                this->visit_expr_wrapper(curr_idx.m_right, true);
                ptr_loads = ptr_loads_copy;
                indices.push_back(tmp);
            }

            ASR::ttype_t* x_mv_type_ = ASRUtils::type_get_past_allocatable(
                ASRUtils::type_get_past_pointer(x_mv_type));
            LCOMPILERS_ASSERT(ASR::is_a<ASR::Array_t>(*x_mv_type_));
            ASR::Array_t* array_t = ASR::down_cast<ASR::Array_t>(x_mv_type_);
            bool is_bindc_array = ASRUtils::expr_abi(x.m_v) == ASR::abiType::BindC;
            if ( LLVM::is_llvm_pointer(*x_mv_type) ||
               ((is_bindc_array && !ASRUtils::is_fixed_size_array(m_dims, n_dims)) &&
                ASR::is_a<ASR::StructInstanceMember_t>(*x.m_v)) ) {
                llvm::Type *array_type = llvm_utils->get_type_from_ttype_t_util(x.m_v, x_mv_type_, module.get());
                array = llvm_utils->CreateLoad2(array_type->getPointerTo(), array);
#if LLVM_VERSION_MAJOR > 16
                ptr_type[array] = array_type;
#endif
            }

            Vec<llvm::Value*> llvm_diminfo;
            llvm_diminfo.reserve(al, 2 * x.n_args + 1);
            if( array_t->m_physical_type == ASR::array_physical_typeType::PointerToDataArray ||
                array_t->m_physical_type == ASR::array_physical_typeType::FixedSizeArray ||
                array_t->m_physical_type == ASR::array_physical_typeType::SIMDArray ||
                (array_t->m_physical_type == ASR::array_physical_typeType::StringArraySinglePointer && ASRUtils::is_fixed_size_array(x_mv_type)) ) {
                int ptr_loads_copy = ptr_loads;
                for( size_t idim = 0; idim < x.n_args; idim++ ) {
                    ptr_loads = 2 - !LLVM::is_llvm_pointer(*ASRUtils::expr_type(m_dims[idim].m_start));
                    this->visit_expr_wrapper(m_dims[idim].m_start, true);
                    llvm::Value* dim_start = tmp;
                    ptr_loads = 2 - !LLVM::is_llvm_pointer(*ASRUtils::expr_type(m_dims[idim].m_length));
                    this->visit_expr_wrapper(m_dims[idim].m_length, true);
                    llvm::Value* dim_size = tmp;
                    llvm_diminfo.push_back(al, dim_start);
                    llvm_diminfo.push_back(al, dim_size);
                }
                ptr_loads = ptr_loads_copy;
            } else if( array_t->m_physical_type == ASR::array_physical_typeType::UnboundedPointerToDataArray ) {
                int ptr_loads_copy = ptr_loads;
                for( size_t idim = 0; idim < x.n_args; idim++ ) {
                    ptr_loads = 2 - !LLVM::is_llvm_pointer(*ASRUtils::expr_type(m_dims[idim].m_start));
                    this->visit_expr_wrapper(m_dims[idim].m_start, true);
                    llvm::Value* dim_start = tmp;
                    llvm_diminfo.push_back(al, dim_start);
                }
                ptr_loads = ptr_loads_copy;
            }
            LCOMPILERS_ASSERT(ASRUtils::extract_n_dims_from_ttype(x_mv_type) > 0);
            bool is_polymorphic = (current_select_type_block_type != nullptr) && ASR::is_a<ASR::Var_t>(*x.m_v) &&
                    (ASRUtils::EXPR2VAR(x.m_v)->m_name == current_selector_var_name);
            if (array_t->m_physical_type == ASR::array_physical_typeType::UnboundedPointerToDataArray) {
                llvm::Type* type = llvm_utils->get_type_from_ttype_t_util(x.m_v, ASRUtils::extract_type(x_mv_type), module.get());
                tmp = arr_descr->get_single_element(type, array, indices, x.n_args, ASRUtils::expr_type(x.m_v),
                                                    true,
                                                    false,
                                                    llvm_diminfo.p, is_polymorphic, current_select_type_block_type,
                                                    true); 
            } else {
                llvm::Type* type;
                bool is_fixed_size = (array_t->m_physical_type == ASR::array_physical_typeType::FixedSizeArray ||
                    array_t->m_physical_type == ASR::array_physical_typeType::SIMDArray ||
                    (
                        array_t->m_physical_type == ASR::array_physical_typeType::StringArraySinglePointer &&
                        ASRUtils::is_fixed_size_array(x_mv_type)
                    )
                );
                if (is_fixed_size) {
                    type = llvm_utils->get_type_from_ttype_t_util(x.m_v, x_mv_type, module.get());
                } else {
                    type = llvm_utils->get_type_from_ttype_t_util(x.m_v, ASRUtils::extract_type(x_mv_type), module.get());
                }
                tmp = arr_descr->get_single_element(type, array, indices, x.n_args, ASRUtils::expr_type(x.m_v),
                                                    array_t->m_physical_type == ASR::array_physical_typeType::PointerToDataArray,
                                                    is_fixed_size, llvm_diminfo.p, is_polymorphic, current_select_type_block_type);
            }
        }
        if( ASR::is_a<ASR::StructType_t>(*ASRUtils::extract_type(x.m_type)) && !ASRUtils::is_class_type(x.m_type) ) {
            current_der_type_name = ASRUtils::symbol_name(
                ASRUtils::symbol_get_past_external(ASRUtils::symbol_get_past_external(ASRUtils::get_struct_sym_from_struct_expr(x.m_v))));
        }
    }

    void visit_ArraySection(const ASR::ArraySection_t& x) {
        if (x.m_value) {
            this->visit_expr_wrapper(x.m_value, true);
            return;
        }
        int64_t ptr_loads_copy = ptr_loads;
        ptr_loads = 0;
        this->visit_expr(*x.m_v);
        ptr_loads = ptr_loads_copy;
        llvm::Value* array = tmp;
        ASR::dimension_t* m_dims;
        [[maybe_unused]] int n_dims = ASRUtils::extract_dimensions_from_ttype(
                        ASRUtils::expr_type(x.m_v), m_dims);
        LCOMPILERS_ASSERT(ASR::is_a<ASR::String_t>(*ASRUtils::expr_type(x.m_v)) &&
                        n_dims == 0);
        // String indexing:
        if (x.n_args == 1) {
            throw CodeGenError("Only string(a:b) supported for now.", x.base.base.loc);
        }

        LCOMPILERS_ASSERT(x.m_args[0].m_left)
        LCOMPILERS_ASSERT(x.m_args[0].m_right)
        //throw CodeGenError("Only string(a:b) for a,b variables for now.", x.base.base.loc);
        // Use the "right" index for now
        this->visit_expr_wrapper(x.m_args[0].m_right, true);
        llvm::Value *idx2 = tmp;
        this->visit_expr_wrapper(x.m_args[0].m_left, true);
        llvm::Value *idx1 = tmp;
        // idx = builder->CreateSub(idx, llvm::ConstantInt::get(context, llvm::APInt(32, 1)));
        //std::vector<llvm::Value*> idx_vec = {llvm::ConstantInt::get(context, llvm::APInt(32, 0)), idx};
        // std::vector<llvm::Value*> idx_vec = {idx};
        llvm::Value *str_data, *str_len;
        std::tie(str_data, str_len) = llvm_utils->get_string_length_data(ASRUtils::get_string_type(x.m_v), array);
        // llvm::Value *p = CreateGEP(str, idx_vec);
        // TODO: Currently the string starts at the right location, but goes to the end of the original string.
        // We have to allocate a new string, copy it and add null termination.
        llvm::Value *step = llvm::ConstantInt::get(context, llvm::APInt(32, 1));
        llvm::Value *present = llvm::ConstantInt::get(context, llvm::APInt(1, 1));
        llvm::Value *p = lfortran_str_slice(str_data, str_len, idx1, idx2, step, present, present);
        tmp = llvm_utils->CreateAlloca(*builder, character_type);
        builder->CreateStore(p, tmp);
    }

    void visit_ArrayReshape(const ASR::ArrayReshape_t& x) {
        this->visit_expr(*x.m_array);
        llvm::Value* array = tmp;
        this->visit_expr(*x.m_shape);
        llvm::Value* shape = tmp;
        ASR::ttype_t* x_m_array_type = ASRUtils::expr_type(x.m_array);
        ASR::array_physical_typeType array_physical_type = ASRUtils::extract_physical_type(x_m_array_type);
        switch( array_physical_type ) {
            case ASR::array_physical_typeType::DescriptorArray: {
                ASR::ttype_t* asr_data_type = ASRUtils::duplicate_type_without_dims(al,
                    ASRUtils::get_contained_type(x_m_array_type), x_m_array_type->base.loc);
                ASR::ttype_t* asr_shape_type = ASRUtils::get_contained_type(ASRUtils::expr_type(x.m_shape));
                llvm::Type* llvm_data_type = llvm_utils->get_type_from_ttype_t_util(x.m_array, asr_data_type, module.get());
#if LLVM_VERSION_MAJOR > 16
                ptr_type[array] = llvm_utils->get_type_from_ttype_t_util(x.m_array,
                    ASRUtils::type_get_past_allocatable_pointer(x_m_array_type), module.get());
                ptr_type[shape] = llvm_utils->get_type_from_ttype_t_util(x.m_shape,
                    ASRUtils::type_get_past_allocatable_pointer(asr_shape_type), module.get());
#endif
                tmp = arr_descr->reshape(array, llvm_data_type, shape, asr_shape_type, module.get());
                break;
            }
            case ASR::array_physical_typeType::FixedSizeArray: {
                llvm::Type* target_type = llvm_utils->get_type_from_ttype_t_util(x.m_array, x_m_array_type, module.get());
                llvm::Value *target = llvm_utils->CreateAlloca(
                    target_type, nullptr, "fixed_size_reshaped_array");
                llvm::Value* target_ = llvm_utils->create_gep2(target_type, target, 0);
                ASR::dimension_t* asr_dims = nullptr;
                size_t asr_n_dims = ASRUtils::extract_dimensions_from_ttype(x_m_array_type, asr_dims);
                int64_t size = ASRUtils::get_fixed_size_of_array(asr_dims, asr_n_dims);
                llvm::Type* llvm_data_type = llvm_utils->get_type_from_ttype_t_util(x.m_array, ASRUtils::type_get_past_array(
                    ASRUtils::type_get_past_allocatable(ASRUtils::type_get_past_pointer(x_m_array_type))), module.get());
                llvm::DataLayout data_layout(module->getDataLayout());
                uint64_t data_size = data_layout.getTypeAllocSize(llvm_data_type);
                llvm::Value* llvm_size = llvm::ConstantInt::get(context, llvm::APInt(32, size));
                llvm_size = builder->CreateMul(llvm_size,
                    llvm::ConstantInt::get(context, llvm::APInt(32, data_size)));
                builder->CreateMemCpy(target_, llvm::MaybeAlign(), array, llvm::MaybeAlign(), llvm_size);
                tmp = target;
                break;
            }
            case ASR::PointerToDataArray: {
                llvm::Type* type_of_array = llvm_utils->get_type_from_ttype_t_util(
                                            x.m_array,
                                            ASRUtils::extract_type(x_m_array_type), module.get());
                int64_t arr_size;
                if(ASRUtils::is_fixed_size_array(x_m_array_type)){
                    arr_size = ASRUtils::get_fixed_size_of_array(x_m_array_type);
                } else {
                    throw LCompilersException("Not Implemented (PointerToDataArray + Not-compile-time size)");
                }

                if(ASRUtils::is_character(*x_m_array_type)){
                    tmp = create_and_setup_string_for_array(x_m_array_type,
                        llvm::ConstantInt::get(context, llvm::APInt(64, arr_size)),
                        false, "StringArrayReshape");
                    builder->CreateMemCpy(
                        llvm_utils->get_stringArray_data(x_m_array_type, tmp), llvm::MaybeAlign(),
                        llvm_utils->get_stringArray_data(ASRUtils::expr_type(x.m_array), array), llvm::MaybeAlign(),
                        llvm_utils->get_stringArray_whole_size(ASRUtils::expr_type(x.m_array))
                    );
                } else {
                    llvm::Value* type_alloca = builder->CreateAlloca(type_of_array,
                                                llvm::ConstantInt::get(context, llvm::APInt(32, arr_size)));
                    llvm::DataLayout data_layout(module->getDataLayout());
                    uint64_t data_size = data_layout.getTypeAllocSize(type_of_array);
                    builder->CreateMemCpy(
                        type_alloca, llvm::MaybeAlign(),
                        array, llvm::MaybeAlign(),
                        llvm::ConstantInt::get(context, llvm::APInt(32, arr_size*data_size)));
                }
                break;
            }
            default: {
                LCOMPILERS_ASSERT(false);
            }
        }
    }

    void visit_ArrayIsContiguous(const ASR::ArrayIsContiguous_t& x) {
        ASR::ttype_t* x_m_array_type = ASRUtils::expr_type(x.m_array);
        llvm::Type* array_type = llvm_utils->get_type_from_ttype_t_util(x.m_array,
                ASRUtils::type_get_past_allocatable(ASRUtils::type_get_past_pointer(x_m_array_type)), module.get());
        int64_t ptr_loads_copy = ptr_loads;
        ptr_loads = 2 - // Sync: instead of 2 - , should this be ptr_loads_copy -
                    (LLVM::is_llvm_pointer(*ASRUtils::expr_type(x.m_array)));
        visit_expr_wrapper(x.m_array);
        ptr_loads = ptr_loads_copy;
        if (is_a<ASR::StructInstanceMember_t>(*x.m_array)) {
            tmp = llvm_utils->CreateLoad2(array_type->getPointerTo(), tmp);
        }
        llvm::Value* llvm_arg1 = tmp;
        ASR::array_physical_typeType physical_type = ASRUtils::extract_physical_type(x_m_array_type);
        switch( physical_type ) {
            case ASR::array_physical_typeType::DescriptorArray: {
                llvm::Value* dim_des_val = arr_descr->get_pointer_to_dimension_descriptor_array(array_type, llvm_arg1);
                llvm::Value* is_contiguous = llvm::ConstantInt::get(context, llvm::APInt(1, 1));
                ASR::dimension_t* m_dims = nullptr;
                int n_dims = ASRUtils::extract_dimensions_from_ttype(x_m_array_type, m_dims);
                llvm::Value* expected_stride = llvm::ConstantInt::get(context, llvm::APInt(32, 1));
                for (int i = 0; i < n_dims; i++) {
                    llvm::Value* dim_index = llvm::ConstantInt::get(context, llvm::APInt(32, i));
                    llvm::Value* dim_desc = arr_descr->get_pointer_to_dimension_descriptor(dim_des_val, dim_index);
                    llvm::Value* dim_start = arr_descr->get_lower_bound(dim_desc);
                    llvm::Value* stride = arr_descr->get_stride(dim_desc);
                    llvm::Value* is_dim_contiguous = builder->CreateICmpEQ(stride, expected_stride);
                    is_contiguous = builder->CreateAnd(is_contiguous, is_dim_contiguous);
                    llvm::Value* dim_size = arr_descr->get_upper_bound(dim_desc);
                    expected_stride = builder->CreateMul(expected_stride, builder->CreateAdd(builder->CreateSub(dim_size, dim_start), llvm::ConstantInt::get(context, llvm::APInt(32, 1))));
                }
                tmp = is_contiguous;
                break;
            }
            case ASR::array_physical_typeType::FixedSizeArray:
            case ASR::array_physical_typeType::SIMDArray:
                tmp = llvm::ConstantInt::get(context, llvm::APInt(1, 1));
                break;
            case ASR::array_physical_typeType::PointerToDataArray:
            case ASR::array_physical_typeType::StringArraySinglePointer:
                tmp = llvm::ConstantInt::get(context, llvm::APInt(1, 0));
                break;
            default:
                LCOMPILERS_ASSERT(false);
        }
    }

    void lookup_EnumValue(const ASR::EnumValue_t& x) {
        ASR::EnumType_t* enum_t = ASR::down_cast<ASR::EnumType_t>(x.m_enum_type);
        ASR::Enum_t* enum_type = ASR::down_cast<ASR::Enum_t>(enum_t->m_enum_type);
        uint32_t h = get_hash((ASR::asr_t*) enum_type);
        llvm::Value* array = llvm_symtab[h];
        tmp = llvm_utils->create_gep(array, tmp);
        tmp = llvm_utils->CreateLoad(llvm_utils->create_gep(tmp, 1));
    }

    void visit_EnumValue(const ASR::EnumValue_t& x) {
        if( x.m_value ) {
            if( ASR::is_a<ASR::Integer_t>(*x.m_type) ) {
                this->visit_expr(*x.m_value);
            } else if( ASR::is_a<ASR::EnumStaticMember_t>(*x.m_v) ) {
                ASR::EnumStaticMember_t* x_enum_member = ASR::down_cast<ASR::EnumStaticMember_t>(x.m_v);
                ASR::Variable_t* x_mv = ASR::down_cast<ASR::Variable_t>(x_enum_member->m_m);
                ASR::EnumType_t* enum_t = ASR::down_cast<ASR::EnumType_t>(x.m_enum_type);
                ASR::Enum_t* enum_type = ASR::down_cast<ASR::Enum_t>(enum_t->m_enum_type);
                for( size_t i = 0; i < enum_type->n_members; i++ ) {
                    if( std::string(enum_type->m_members[i]) == std::string(x_mv->m_name) ) {
                        tmp = llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), llvm::APInt(32, i));
                        break ;
                    }
                }
                if( lookup_enum_value_for_nonints ) {
                    lookup_EnumValue(x);
                }
            }
            return ;
        }

        visit_expr(*x.m_v);
        if( ASR::is_a<ASR::StructInstanceMember_t>(*x.m_v) ) {
            tmp = llvm_utils->CreateLoad(tmp);
        }
        if( !ASR::is_a<ASR::Integer_t>(*x.m_type) && lookup_enum_value_for_nonints ) {
            lookup_EnumValue(x);
        }
    }

    void visit_EnumName(const ASR::EnumName_t& x) {
        if( x.m_value ) {
            this->visit_expr(*x.m_value);
            return ;
        }

        visit_expr(*x.m_v);
        if( ASR::is_a<ASR::StructInstanceMember_t>(*x.m_v) ) {
            tmp = llvm_utils->CreateLoad(tmp);
        }
        ASR::EnumType_t* enum_t = ASR::down_cast<ASR::EnumType_t>(x.m_enum_type);
        ASR::Enum_t* enum_type = ASR::down_cast<ASR::Enum_t>(enum_t->m_enum_type);
        uint32_t h = get_hash((ASR::asr_t*) enum_type);
        llvm::Value* array = llvm_symtab[h];
        if( ASR::is_a<ASR::Integer_t>(*enum_type->m_type) ) {
            int64_t min_value = INT64_MAX;

            for( auto itr: enum_type->m_symtab->get_scope() ) {
                ASR::Variable_t* itr_var = ASR::down_cast<ASR::Variable_t>(itr.second);
                ASR::expr_t* value = ASRUtils::expr_value(itr_var->m_symbolic_value);
                int64_t value_int64 = -1;
                ASRUtils::extract_value(value, value_int64);
                min_value = std::min(value_int64, min_value);
            }
            tmp = builder->CreateSub(tmp, llvm::ConstantInt::get(tmp->getType(),
                        llvm::APInt(32, min_value, true)));
            tmp = llvm_utils->create_gep(array, tmp);
            tmp = llvm_utils->create_gep(tmp, 0);
        }
    }

    void visit_EnumConstructor(const ASR::EnumConstructor_t& x) {
        LCOMPILERS_ASSERT(x.n_args == 1);
        ASR::expr_t* m_arg = x.m_args[0];
        this->visit_expr(*m_arg);
    }

    void visit_UnionConstructor([[maybe_unused]] const ASR::UnionConstructor_t& x) {
        LCOMPILERS_ASSERT(x.n_args == 0);
    }

    llvm::Value* SizeOfTypeUtil(ASR::expr_t* expr, ASR::ttype_t* m_type, llvm::Type* output_type,
        ASR::ttype_t* output_type_asr) {
        llvm::Type* llvm_type = llvm_utils->get_type_from_ttype_t_util(expr, m_type, module.get());
        llvm::DataLayout data_layout(module->getDataLayout());
        int64_t type_size = data_layout.getTypeAllocSize(llvm_type);
        return llvm::ConstantInt::get(output_type, llvm::APInt(
            ASRUtils::extract_kind_from_ttype_t(output_type_asr) * 8, type_size));
    }

    void visit_StructInstanceMember(const ASR::StructInstanceMember_t& x) {
        if (x.m_value) {
            this->visit_expr_wrapper(x.m_value, true);
            return;
        }
        current_der_type_name = "";
        ASR::ttype_t* x_m_v_type = ASRUtils::expr_type(x.m_v);
        int64_t ptr_loads_copy = ptr_loads;
        if( ASR::is_a<ASR::UnionInstanceMember_t>(*x.m_v) ||
            ASRUtils::is_class_type(ASRUtils::extract_type(x_m_v_type)) ) {
            ptr_loads = 0;
        } else {
            ptr_loads = 2 - LLVM::is_llvm_pointer(*x_m_v_type);
        }
        this->visit_expr(*x.m_v);
        ptr_loads = ptr_loads_copy;
        if (ASRUtils::is_unlimited_polymorphic_type(x.m_v)) {
            if( current_select_type_block_type ) {
                current_der_type_name = current_select_type_block_der_type;
            }
        } else if (ASRUtils::is_class_type(ASRUtils::type_get_past_pointer(
                ASRUtils::type_get_past_allocatable(x_m_v_type))) ) {
            if (ASRUtils::is_allocatable(x_m_v_type)) {
                ASR::ttype_t* wrapped_struct_type = ASRUtils::make_StructType_t_util(al, x_m_v_type->base.loc,
                                ASRUtils::symbol_get_past_external(ASRUtils::get_struct_sym_from_struct_expr(x.m_v)));
                llvm::Type* wrapper_struct_llvm_type = llvm_utils->get_type_from_ttype_t_util(x.m_v, wrapped_struct_type, module.get())->getPointerTo();
                llvm::Type* x_mv_llvm_type = llvm_utils->get_type_from_ttype_t_util(x.m_v, x_m_v_type, module.get());
                tmp = llvm_utils->create_gep2(x_mv_llvm_type, tmp, 1);
                tmp = llvm_utils->CreateLoad2(wrapper_struct_llvm_type, tmp);
            } else {
                ASR::ttype_t* x_m_v_type_ = ASRUtils::type_get_past_allocatable(
                    ASRUtils::type_get_past_pointer(x_m_v_type));
                llvm::Type* type = llvm_utils->get_type_from_ttype_t_util(x.m_v, x_m_v_type_, module.get());
                tmp = llvm_utils->CreateLoad2(
                    name2dertype[current_der_type_name]->getPointerTo(), llvm_utils->create_gep2(type, tmp, 1));
            }
            if( current_select_type_block_type && ASR::is_a<ASR::Var_t>(*x.m_v) &&
                (ASRUtils::EXPR2VAR(x.m_v)->m_name == current_selector_var_name) ) {
                tmp = builder->CreateBitCast(tmp, current_select_type_block_type->getPointerTo());
                current_der_type_name = current_select_type_block_der_type;
            } else {
                // TODO: Select type by comparing with vtab
            }
        }
        ASR::Variable_t* member = down_cast<ASR::Variable_t>(symbol_get_past_external(x.m_m));
        std::string member_name = std::string(member->m_name);
        LCOMPILERS_ASSERT(current_der_type_name.size() != 0);
        while( name2memidx[current_der_type_name].find(member_name) == name2memidx[current_der_type_name].end() ) {
            if( dertype2parent.find(current_der_type_name) == dertype2parent.end() ) {
                throw CodeGenError(current_der_type_name + " doesn't have any member named " + member_name,
                                    x.base.base.loc);
            }
            tmp = llvm_utils->create_gep2(name2dertype[current_der_type_name], tmp, 0);
            current_der_type_name = dertype2parent[current_der_type_name];
        }
        int member_idx = name2memidx[current_der_type_name][member_name];

        llvm::Type *xtype = name2dertype[current_der_type_name];
        if ((ASRUtils::is_allocatable(x_m_v_type) || ASRUtils::is_pointer(x_m_v_type)) &&
            ASR::is_a<ASR::StructInstanceMember_t>(*x.m_v) &&
            !ASRUtils::is_class_type(ASRUtils::extract_type(x_m_v_type))) {
            tmp = llvm_utils->CreateLoad2(xtype->getPointerTo(), tmp);
        }
        tmp = llvm_utils->create_gep2(xtype, tmp, member_idx);
        ASR::ttype_t* member_type = ASRUtils::type_get_past_pointer(
            ASRUtils::type_get_past_allocatable(member->m_type));
#if LLVM_VERSION_MAJOR > 16
        ptr_type[tmp] = llvm_utils->get_type_from_ttype_t_util(ASRUtils::get_expr_from_sym(al, x.m_m),
            member_type, module.get());
#endif
        if( ASR::is_a<ASR::StructType_t>(*member_type) ) {
            ASR::symbol_t *s_sym = member->m_type_declaration;
            current_der_type_name = ASRUtils::symbol_name(
                ASRUtils::symbol_get_past_external(s_sym));
            if (!ASRUtils::is_class_type(member_type)) {
                uint32_t h = get_hash((ASR::asr_t*)member);
                if( llvm_symtab.find(h) != llvm_symtab.end() ) {
                    tmp = llvm_symtab[h];
                }
            }
        }
    }

    void visit_StructConstant(const ASR::StructConstant_t& x) {
        std::vector<llvm::Constant *> elements;
        llvm::StructType* t = llvm::cast<llvm::StructType>(
            llvm_utils->getStructType(ASR::down_cast<ASR::Struct_t>(ASRUtils::symbol_get_past_external(x.m_dt_sym)), module.get()));
        ASR::Struct_t* struct_
            = ASR::down_cast<ASR::Struct_t>(ASRUtils::symbol_get_past_external(x.m_dt_sym));

        LCOMPILERS_ASSERT(x.n_args == struct_->n_members);
        for (size_t i = 0; i < x.n_args; ++i) {
            ASR::expr_t *value = x.m_args[i].m_value;
            llvm::Constant* initializer = nullptr;
            llvm::Type* type = nullptr;
            if (value == nullptr) {
                auto member2sym = struct_->m_symtab->get_scope();
                LCOMPILERS_ASSERT(member2sym[struct_->m_members[i]]->type == ASR::symbolType::Variable);
                ASR::Variable_t *s = ASR::down_cast<ASR::Variable_t>(member2sym[struct_->m_members[i]]);
                type = llvm_utils->get_type_from_ttype_t_util(
                    ASRUtils::EXPR(ASR::make_Var_t(al, s->base.base.loc, &s->base)),
                    s->m_type,
                    module.get());
                if (type->isArrayTy()) {
                    initializer = llvm::ConstantArray::getNullValue(type);
                } else {
                    initializer = llvm::Constant::getNullValue(type);
                }
            } else if (ASR::is_a<ASR::ArrayConstant_t>(*value)) {
                ASR::ArrayConstant_t *arr_expr = ASR::down_cast<ASR::ArrayConstant_t>(value);
                type = llvm_utils->get_type_from_ttype_t_util(value, arr_expr->m_type, module.get());
                initializer = get_const_array(value, type);
            } else {
                visit_expr_wrapper(value);
                initializer = llvm::dyn_cast<llvm::Constant>(tmp);
                if (!initializer) {
                    throw CodeGenError("Non-constant value found in struct initialization");
                }
            }
            elements.push_back(initializer);
        }
        tmp = llvm::ConstantStruct::get(t, elements);
        current_der_type_name = ASRUtils::symbol_name(x.m_dt_sym);
    }

    llvm::Constant* get_const_array(ASR::expr_t *value, llvm::Type* type) {
        LCOMPILERS_ASSERT(ASR::is_a<ASR::ArrayConstant_t>(*value));
        ASR::ArrayConstant_t* arr_const = ASR::down_cast<ASR::ArrayConstant_t>(value);
        std::vector<llvm::Constant*> arr_elements;
        size_t arr_const_size = (size_t) ASRUtils::get_fixed_size_of_array(arr_const->m_type);
        arr_elements.reserve(arr_const_size);
        int a_kind;
        for (size_t i = 0; i < arr_const_size; i++) {
            ASR::expr_t* elem = ASRUtils::fetch_ArrayConstant_value(al, arr_const, i);
            a_kind = ASRUtils::extract_kind_from_ttype_t(ASRUtils::expr_type(elem));
            if (ASR::is_a<ASR::IntegerConstant_t>(*elem)) {
                ASR::IntegerConstant_t* int_const = ASR::down_cast<ASR::IntegerConstant_t>(elem);
                arr_elements.push_back(llvm::ConstantInt::get(
                    context, llvm::APInt(8 * a_kind, int_const->m_n, true)));
            } else if (ASR::is_a<ASR::RealConstant_t>(*elem)) {
                ASR::RealConstant_t* real_const = ASR::down_cast<ASR::RealConstant_t>(elem);
                if (a_kind == 4) {
                    arr_elements.push_back(llvm::ConstantFP::get(
                        context, llvm::APFloat((float) real_const->m_r)));
                } else if (a_kind == 8) {
                    arr_elements.push_back(llvm::ConstantFP::get(
                        context, llvm::APFloat((double) real_const->m_r)));
                }
            } else if (ASR::is_a<ASR::LogicalConstant_t>(*elem)) {
                ASR::LogicalConstant_t* logical_const = ASR::down_cast<ASR::LogicalConstant_t>(elem);
                arr_elements.push_back(llvm::ConstantInt::get(
                    context, llvm::APInt(1, logical_const->m_value)));
            }
        }
        llvm::ArrayType* arr_type = llvm::ArrayType::get(type, arr_const_size);
        llvm::Constant* initializer = nullptr;
        if (isNullValueArray(arr_elements)) {
            initializer = llvm::ConstantArray::getNullValue(type);
        } else {
            initializer = llvm::ConstantArray::get(arr_type, arr_elements);
        }
        return initializer;
    }

    bool isNullValueArray(const std::vector<llvm::Constant*>& elements) {
        return std::all_of(elements.begin(), elements.end(),
            [](llvm::Constant* elem) { return elem->isNullValue(); });
    }

    void set_global_variable_linkage_as_common(llvm::Value* ptr, ASR::abiType x_abi) {
        if ( compiler_options.separate_compilation ) {
            /*
                In case of global variables without initialization, clang
                generates a common symbol. For the following C code:

                ```
                int global_var;
                int global_variable_initalised = 42;
                ```

                on using `clang -S -emit-llvm` we get:

                ```
                ; ModuleID = 'a.c'
                source_filename = "a.c"
                target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
                target triple = "arm64-apple-macosx14.0.0"

                @global_var_initialized = global i32 42, align 4
                @global_var = common global i32 0, align 4

                !llvm.module.flags = !{!0, !1, !2, !3, !4}
                !llvm.ident = !{!5}

                !0 = !{i32 2, !"SDK Version", [2 x i32] [i32 14, i32 4]}
                !1 = !{i32 1, !"wchar_size", i32 4}
                !2 = !{i32 8, !"PIC Level", i32 2}
                !3 = !{i32 7, !"uwtable", i32 1}
                !4 = !{i32 7, !"frame-pointer", i32 1}
                !5 = !{!"Apple clang version 15.0.0 (clang-1500.3.9.4)"}
                ```

                Hence, we set the linkage to CommonLinkage for
                global variables without initialization in case
                `compiler_options.separate_compilation` is set to true.
            */
            llvm::GlobalVariable *gv = llvm::cast<llvm::GlobalVariable>(
                ptr
            );
            if ( x_abi != ASR::abiType::ExternalUndefined &&
                x_abi != ASR::abiType::BindC ) {
                gv->setLinkage(llvm::GlobalValue::CommonLinkage);
            }
        }
    }

    void visit_Variable(const ASR::Variable_t &x) {
        if (x.m_value && x.m_storage == ASR::storage_typeType::Parameter) {
            this->visit_expr_wrapper(x.m_value, true);
            return;
        }
        uint32_t h = get_hash((ASR::asr_t*)&x);
        // This happens at global scope, so the intent can only be either local
        // (global variable declared/initialized in this translation unit), or
        // external (global variable not declared/initialized in this
        // translation unit, just referenced).
        LCOMPILERS_ASSERT(x.m_intent == intent_local || x.m_intent == ASRUtils::intent_unspecified
            || x.m_abi == ASR::abiType::ExternalUndefined);
        bool external = (x.m_abi != ASR::abiType::Source);
        llvm::Constant* init_value = nullptr;
        if (x.m_symbolic_value != nullptr && 
            !ASRUtils::is_string_only(x.m_type)){
            this->visit_expr_wrapper(x.m_symbolic_value, true);
            init_value = llvm::dyn_cast<llvm::Constant>(tmp);
        }

        // variable name to use in declaration
        std::string llvm_var_name = "";
        if (x.m_abi == ASR::abiType::BindC && x.m_bindc_name) {
            // for external global variable with bindc, use the C name
            llvm_var_name = x.m_bindc_name;
        } else {
            if ( !( ASRUtils::is_struct(*x.m_type) || ASRUtils::is_class_type(x.m_type) ) ) {
                llvm_var_name = mangle_prefix + x.m_name;
            } else {
                llvm_var_name = x.m_name;
            }
        }
        if (x.m_type->type == ASR::ttypeType::Integer
            || x.m_type->type == ASR::ttypeType::UnsignedInteger) {
            int a_kind = ASRUtils::extract_kind_from_ttype_t(x.m_type);
            llvm::Type *type;
            int init_value_bits = 8*a_kind;
            type = llvm_utils->getIntType(a_kind);
            llvm::Constant *ptr = module->getOrInsertGlobal(llvm_var_name,
                type);
            if (!external) {
                if (ASRUtils::is_array(x.m_type)) {
                    throw CodeGenError("Arrays are not supported by visit_Variable");
                }
                if (init_value) {
                    module->getNamedGlobal(llvm_var_name)->setInitializer(
                            init_value);
                } else {
                    module->getNamedGlobal(llvm_var_name)->setInitializer(
                            llvm::ConstantInt::get(context,
                                llvm::APInt(init_value_bits, 0)));
                    set_global_variable_linkage_as_common(ptr, x.m_abi);
                }
            }
            llvm_symtab[h] = ptr;
        } else if (x.m_type->type == ASR::ttypeType::Real) {
            int a_kind = down_cast<ASR::Real_t>(x.m_type)->m_kind;
            llvm::Type *type;
            int init_value_bits = 8*a_kind;
            type = llvm_utils->getFPType(a_kind);
            llvm::Constant *ptr = module->getOrInsertGlobal(llvm_var_name, type);
            if (!external) {
                if (init_value) {
                    module->getNamedGlobal(llvm_var_name)->setInitializer(
                            init_value);
                } else {
                    if( init_value_bits == 32 ) {
                        module->getNamedGlobal(llvm_var_name)->setInitializer(
                                llvm::ConstantFP::get(context,
                                    llvm::APFloat((float)0)));
                    } else if( init_value_bits == 64 ) {
                        module->getNamedGlobal(llvm_var_name)->setInitializer(
                                llvm::ConstantFP::get(context,
                                    llvm::APFloat((double)0)));
                    }
                    set_global_variable_linkage_as_common(ptr, x.m_abi);
                }
            }
            llvm_symtab[h] = ptr;
        } else if (x.m_type->type == ASR::ttypeType::Array) {
            llvm::Constant *ptr{};
            if(ASRUtils::is_character(*x.m_type)){
                ASR::expr_t* value = nullptr;
                if( x.m_value ) {
                    value = x.m_value;
                } else if( x.m_symbolic_value &&
                        ASRUtils::is_value_constant(x.m_symbolic_value) ) {
                    value = x.m_symbolic_value;
                }
                if(value){LCOMPILERS_ASSERT(ASR::is_a<ASR::ArrayConstant_t>(*value));}
                ptr = llvm::cast<llvm::Constant>(llvm_utils->handle_global_nonallocatable_stringArray(al, 
                    ASR::down_cast<ASR::Array_t>(x.m_type), value?ASR::down_cast<ASR::ArrayConstant_t>(value):nullptr, llvm_var_name));

            } else {
                llvm::Type* type = llvm_utils->get_type_from_ttype_t_util(ASRUtils::EXPR(
                    ASR::make_Var_t(al, x.base.base.loc, const_cast<ASR::symbol_t*>(&x.base))),
                    x.m_type, module.get());
                ptr = module->getOrInsertGlobal(llvm_var_name, type);
                if (!external) {
                    ASR::expr_t* value = nullptr;
                    if( x.m_value ) {
                        value = x.m_value;
                    } else if( x.m_symbolic_value &&
                            ASRUtils::is_value_constant(x.m_symbolic_value) ) {
                        value = x.m_symbolic_value;
                    }
                    if (value) {
                        llvm::Constant* initializer = get_const_array(value, type);
                        module->getNamedGlobal(llvm_var_name)->setInitializer(initializer);
                    } else {
                        module->getNamedGlobal(llvm_var_name)->setInitializer(llvm::ConstantArray::getNullValue(type));
                        set_global_variable_linkage_as_common(ptr, x.m_abi);
                    }
                }
            }
            llvm_symtab[h] = ptr;
        } else if (x.m_type->type == ASR::ttypeType::Logical) {
            llvm::Constant *ptr = module->getOrInsertGlobal(llvm_var_name,
                llvm::Type::getInt1Ty(context));
            if (!external) {
                if (init_value) {
                    module->getNamedGlobal(llvm_var_name)->setInitializer(
                            init_value);
                } else {
                    module->getNamedGlobal(llvm_var_name)->setInitializer(
                            llvm::ConstantInt::get(context,
                                llvm::APInt(1, 0)));
                    set_global_variable_linkage_as_common(ptr, x.m_abi);
                }
            }
            llvm_symtab[h] = ptr;
        } else if (x.m_type->type == ASR::ttypeType::String) {
            ASR::String_t* str = ASRUtils::get_string_type(x.m_type);
            llvm::Value *ptr{};
                bool is_const = (x.m_storage == ASR::storage_typeType::Parameter);
                if (!external) {
                    std::string initial_string_value;
                    if (x.m_symbolic_value) {
                        ASR::StringConstant_t* str_const = ASR::down_cast<ASR::StringConstant_t>(ASRUtils::expr_value(x.m_symbolic_value));
                        int64_t len; ASRUtils::extract_value(ASRUtils::get_string_type(str_const->m_type)->m_len, len);
                        initial_string_value = std::string(str_const->m_s, len);
                    } else {
                        initial_string_value = "";
                    }
                    ptr = llvm_utils->declare_global_string(str, initial_string_value,
                        is_const, llvm_var_name,
                        compiler_options.separate_compilation ? llvm::GlobalValue::ExternalLinkage : llvm::GlobalValue::PrivateLinkage);
                } else {
                    ptr = module->getOrInsertGlobal(llvm_var_name, llvm_utils->get_StringType(x.m_type));
                }
            llvm_symtab[h] = ptr;
        } else if( x.m_type->type == ASR::ttypeType::CPtr ) {
            llvm::Type* void_ptr = llvm::Type::getVoidTy(context)->getPointerTo();
            llvm::Constant *ptr = module->getOrInsertGlobal(llvm_var_name,
                void_ptr);
            if (!external) {
                if (init_value) {
                    module->getNamedGlobal(llvm_var_name)->setInitializer(
                            init_value);
                } else {
                    module->getNamedGlobal(llvm_var_name)->setInitializer(
                            llvm::ConstantPointerNull::get(
                                static_cast<llvm::PointerType*>(void_ptr))
                            );
                    set_global_variable_linkage_as_common(ptr, x.m_abi);
                }
            }
            llvm_symtab[h] = ptr;
        } else if( x.m_type->type == ASR::ttypeType::StructType ) {
            ASR::StructType_t* struct_t = ASR::down_cast<ASR::StructType_t>(x.m_type);
            if (init_value == nullptr && x.m_type->type == ASR::ttypeType::StructType) {
                ASR::Struct_t* struct_sym = ASR::down_cast<ASR::Struct_t>(ASRUtils::symbol_get_past_external(x.m_type_declaration));
                std::vector<llvm::Constant*> field_values;
                for (auto& member : struct_sym->m_symtab->get_scope()) {
                    ASR::symbol_t* sym = member.second;
                    if (!ASR::is_a<ASR::Variable_t>(*sym))
                        continue;
                    ASR::Variable_t* var = ASR::down_cast<ASR::Variable_t>(sym);
                    if (var->m_value != nullptr) {
                        llvm::Constant* c = llvm_utils->create_llvm_constant_from_asr_expr(var->m_value, module.get());
                        field_values.push_back(c);
                    } else {
                        llvm::Type* member_type = llvm_utils->get_type_from_ttype_t_util(ASRUtils::EXPR(
                            ASR::make_Var_t(al, var->base.base.loc, const_cast<ASR::symbol_t*>(&var->base))), 
                            var->m_type, module.get());
                        field_values.push_back(llvm::Constant::getNullValue(member_type));
                    }
                }
                llvm::StructType* llvm_struct_type = llvm::cast<llvm::StructType>(llvm_utils->get_type_from_ttype_t_util(ASRUtils::EXPR(
                    ASR::make_Var_t(al, x.base.base.loc, const_cast<ASR::symbol_t*>(&x.base))), x.m_type, module.get()));
                init_value = llvm::ConstantStruct::get(llvm_struct_type, field_values);
            }
            if (struct_t->m_is_cstruct) {
                if( x.m_type_declaration && ASRUtils::is_c_ptr(x.m_type_declaration) ) {
                    llvm::Type* void_ptr = llvm::Type::getVoidTy(context)->getPointerTo();
                    llvm::Constant *ptr = module->getOrInsertGlobal(llvm_var_name,
                        void_ptr);
                    if (!external) {
                        if (init_value) {
                            module->getNamedGlobal(llvm_var_name)->setInitializer(
                                    init_value);
                        } else {
                            module->getNamedGlobal(llvm_var_name)->setInitializer(
                                    llvm::ConstantPointerNull::get(
                                        static_cast<llvm::PointerType*>(void_ptr))
                                    );
                            set_global_variable_linkage_as_common(ptr, x.m_abi);
                        }
                    }
                    llvm_symtab[h] = ptr;
                } else {
                    llvm::Type* type = llvm_utils->get_type_from_ttype_t_util(ASRUtils::EXPR(
                    ASR::make_Var_t(al, x.base.base.loc, const_cast<ASR::symbol_t*>(&x.base))), x.m_type, module.get());
                    llvm::Constant *ptr = module->getOrInsertGlobal(llvm_var_name,
                        type);
                    if (!external) {
                        if (init_value) {
                            module->getNamedGlobal(llvm_var_name)->setInitializer(
                                    init_value);
                        } else {
                            module->getNamedGlobal(llvm_var_name)->setInitializer(
                                    llvm::Constant::getNullValue(type)
                                );
                            set_global_variable_linkage_as_common(ptr, x.m_abi);
                        }
                    }
                    llvm_symtab[h] = ptr;
                }
            } else {
                llvm::Type* type = llvm_utils->get_type_from_ttype_t_util(
                    ASRUtils::EXPR(ASR::make_Var_t(al, x.base.base.loc, x.m_type_declaration)),
                    x.m_type, module.get());
                llvm::Constant *ptr = module->getOrInsertGlobal(llvm_var_name,
                    type);
                if (!external) {
                    if (init_value) {
                        module->getNamedGlobal(llvm_var_name)->setInitializer(
                                init_value);
                    } else {
                        module->getNamedGlobal(llvm_var_name)
                            ->setInitializer(llvm::Constant::getNullValue(type));
                        set_global_variable_linkage_as_common(ptr, x.m_abi);
                    }
                }
                llvm_symtab[h] = ptr;
            }
        } else if(x.m_type->type == ASR::ttypeType::Pointer ||
                    x.m_type->type == ASR::ttypeType::Allocatable) {
            ASR::dimension_t* m_dims = nullptr;
            int n_dims = 0, a_kind = -1;
            bool is_array_type = false, is_malloc_array_type = false, is_list = false;
            llvm::Type* x_ptr = llvm_utils->get_type_from_ttype_t(
                ASRUtils::EXPR(ASR::make_Var_t(
                    al, x.base.base.loc, const_cast<ASR::symbol_t*>(&x.base))), x.m_type,
                    x.m_type_declaration, x.m_storage, is_array_type,
                    is_malloc_array_type, is_list, m_dims, n_dims, a_kind, module.get());
            llvm::Constant *ptr = module->getOrInsertGlobal(llvm_var_name,
                x_ptr);
            llvm::Type* type_ = llvm_utils->get_type_from_ttype_t_util(ASRUtils::EXPR(
                ASR::make_Var_t(al, x.base.base.loc, const_cast<ASR::symbol_t*>(&x.base))),
                ASRUtils::type_get_past_pointer(
                ASRUtils::type_get_past_allocatable(x.m_type)),
                module.get(), x.m_abi);
#if LLVM_VERSION_MAJOR > 16
            if ( LLVM::is_llvm_pointer(*x.m_type) &&
                    ASR::is_a<ASR::StructType_t>(*ASRUtils::type_get_past_pointer(
                    ASRUtils::type_get_past_allocatable(x.m_type))) && !ASRUtils::is_class_type(ASRUtils::type_get_past_pointer(
                    ASRUtils::type_get_past_allocatable(x.m_type)))) {
                ptr_type[ptr] = type_->getPointerTo();
            } else {
                ptr_type[ptr] = type_;
            }
#endif
            if (ASRUtils::is_array(x.m_type)) {  // memorize arrays only.
                allocatable_array_details.push_back(
                    { ASRUtils::EXPR(ASR::make_Var_t(
                          al, x.base.base.loc, const_cast<ASR::symbol_t*>(&x.base))),
                      ptr,
                      type_,
                      x.m_type,
                      ASRUtils::extract_dimensions_from_ttype(x.m_type, m_dims) });
            } else if (ASRUtils::is_character(*x.m_type) && !init_value) {
                // set all members of string_descriptor to null and zeroes.
                init_value = llvm::ConstantAggregateZero::get(string_descriptor);
            }
            if (!external) {
                if (init_value) {
                    module->getNamedGlobal(llvm_var_name)->setInitializer(
                            init_value);
                } else {
                    module->getNamedGlobal(llvm_var_name)->setInitializer(
                            llvm::ConstantPointerNull::get(
                                static_cast<llvm::PointerType*>(x_ptr))
                            );
                    set_global_variable_linkage_as_common(ptr, x.m_abi);
                }
            }
            llvm_symtab[h] = ptr;
        } else if (x.m_type->type == ASR::ttypeType::List) {
            llvm::StructType* list_type = static_cast<llvm::StructType*>(
                llvm_utils->get_type_from_ttype_t_util(ASRUtils::EXPR(
                ASR::make_Var_t(al, x.base.base.loc, const_cast<ASR::symbol_t*>(&x.base))), x.m_type, module.get()));
            llvm::Constant *ptr = module->getOrInsertGlobal(llvm_var_name, list_type);
            module->getNamedGlobal(llvm_var_name)->setInitializer(
                llvm::ConstantStruct::get(list_type,
                llvm::Constant::getNullValue(list_type)));
            llvm_symtab[h] = ptr;
        } else if (x.m_type->type == ASR::ttypeType::Tuple) {
            llvm::StructType* tuple_type = static_cast<llvm::StructType*>(
                llvm_utils->get_type_from_ttype_t_util(ASRUtils::EXPR(
                ASR::make_Var_t(al, x.base.base.loc, const_cast<ASR::symbol_t*>(&x.base))), x.m_type, module.get()));
            llvm::Constant *ptr = module->getOrInsertGlobal(llvm_var_name, tuple_type);
            module->getNamedGlobal(llvm_var_name)->setInitializer(
                llvm::ConstantStruct::get(tuple_type,
                llvm::Constant::getNullValue(tuple_type)));
            llvm_symtab[h] = ptr;
        } else if(x.m_type->type == ASR::ttypeType::Dict) {
            llvm::StructType* dict_type = static_cast<llvm::StructType*>(
                llvm_utils->get_type_from_ttype_t_util(ASRUtils::EXPR(
                ASR::make_Var_t(al, x.base.base.loc, const_cast<ASR::symbol_t*>(&x.base))), x.m_type, module.get()));
            llvm::Constant *ptr = module->getOrInsertGlobal(llvm_var_name, dict_type);
            module->getNamedGlobal(llvm_var_name)->setInitializer(
                llvm::ConstantStruct::get(dict_type,
                llvm::Constant::getNullValue(dict_type)));
            llvm_symtab[h] = ptr;
        } else if(x.m_type->type == ASR::ttypeType::Set) {
            llvm::StructType* set_type = static_cast<llvm::StructType*>(
                llvm_utils->get_type_from_ttype_t_util(ASRUtils::EXPR(
                ASR::make_Var_t(al, x.base.base.loc, const_cast<ASR::symbol_t*>(&x.base))), x.m_type, module.get()));
            llvm::Constant *ptr = module->getOrInsertGlobal(llvm_var_name, set_type);
            module->getNamedGlobal(llvm_var_name)->setInitializer(
                llvm::ConstantStruct::get(set_type,
                llvm::Constant::getNullValue(set_type)));
            llvm_symtab[h] = ptr;
        } else if (x.m_type->type == ASR::ttypeType::Complex) {
            int a_kind = ASRUtils::extract_kind_from_ttype_t(x.m_type);

            llvm::Constant* re;
            llvm::Constant* im;
            llvm::Type* type = llvm_utils->get_type_from_ttype_t_util(ASRUtils::EXPR(
                ASR::make_Var_t(al, x.base.base.loc, const_cast<ASR::symbol_t*>(&x.base))), x.m_type, module.get());;
            llvm::Constant* ptr = module->getOrInsertGlobal(llvm_var_name, type);

            if (!external) {
                double x_re = 0.0, x_im = 0.0;
                if (x.m_value) {
                    LCOMPILERS_ASSERT(ASR::is_a<ASR::ComplexConstant_t>(*x.m_value));
                    ASR::ComplexConstant_t* x_cc = ASR::down_cast<ASR::ComplexConstant_t>(x.m_value);
                    x_re = x_cc->m_re; x_im = x_cc->m_im;
                }
                if (init_value) {
                    module->getNamedGlobal(llvm_var_name)->setInitializer(init_value);
                } else {
                    switch (a_kind) {
                        case 4: {
                            re = llvm::ConstantFP::get(context, llvm::APFloat((float) x_re));
                            im = llvm::ConstantFP::get(context, llvm::APFloat((float) x_im));
                            type = complex_type_4;
                            break;
                        }
                        case 8: {
                            re = llvm::ConstantFP::get(context, llvm::APFloat((double) x_re));
                            im = llvm::ConstantFP::get(context, llvm::APFloat((double) x_im));
                            type = complex_type_8;
                            break;
                        }
                        default: {
                            throw CodeGenError("kind type is not supported");
                        }
                    }
                    // Create a constant structure to represent the complex number
                    std::vector<llvm::Constant*> elements = { re, im };
                    llvm::Constant* complex_init = llvm::ConstantStruct::get(static_cast<llvm::StructType*>(type), elements);
                    module->getNamedGlobal(llvm_var_name)->setInitializer(complex_init);
                }
            }
            llvm_symtab[h] = ptr;
        } else if (x.m_type->type == ASR::ttypeType::TypeParameter) {
            // Ignore type variables
        } else if (x.m_type->type == ASR::ttypeType::FunctionType) {
            llvm::Type* type = llvm_utils->get_type_from_ttype_t_util(ASRUtils::EXPR(
                ASR::make_Var_t(al, x.base.base.loc, const_cast<ASR::symbol_t*>(&x.base))), x.m_type, module.get());
            llvm::Constant *ptr = module->getOrInsertGlobal(llvm_var_name, type);
            if (!external) {
                if (init_value) {
                    module->getNamedGlobal(llvm_var_name)->setInitializer(
                            init_value);
                } else {
                    module->getNamedGlobal(llvm_var_name)->setInitializer(
                            llvm::Constant::getNullValue(type)
                        );
                    set_global_variable_linkage_as_common(ptr, x.m_abi);
                }
            }
            llvm_symtab[h] = ptr;
#if LLVM_VERSION_MAJOR > 16
            ptr_type[ptr] = type;
#endif
        } else {
            throw CodeGenError("Variable type not supported " + ASRUtils::type_to_str_python(x.m_type), x.base.base.loc);
        }
    }

    void visit_PointerNullConstant(const ASR::PointerNullConstant_t& x){
        llvm::Type* value_type;

        value_type = ASRUtils::is_array(x.m_type)?
            llvm_utils->get_type_from_ttype_t_util(x.m_var_expr,
                ASRUtils::extract_type(x.m_type), module.get())->getPointerTo():
            llvm_utils->get_type_from_ttype_t_util(x.m_var_expr, x.m_type, module.get());
        tmp = llvm::ConstantPointerNull::get(static_cast<llvm::PointerType*>(value_type));
    }

    void visit_Enum(const ASR::Enum_t& x) {
        if ( x.m_abi == ASR::abiType::ExternalUndefined ) {
            return;
        }
        if( x.m_enum_value_type == ASR::enumtypeType::IntegerUnique &&
            x.m_abi == ASR::abiType::BindC ) {
            throw CodeGenError("C-interoperation support for non-consecutive but uniquely "
                               "valued integer enums isn't available yet.");
        }
        bool is_integer = ASR::is_a<ASR::Integer_t>(*x.m_type);
        ASR::storage_typeType m_storage = ASR::storage_typeType::Default;
        bool is_array_type = false, is_malloc_array_type = false, is_list = false;
        ASR::dimension_t* m_dims = nullptr;
        int n_dims = -1, a_kind = -1;
        llvm::Type* value_type = llvm_utils->get_type_from_ttype_t(ASRUtils::EXPR(ASR::make_Var_t(
                    al, x.base.base.loc, const_cast<ASR::symbol_t*>(&x.base))),
            x.m_type, nullptr, m_storage, is_array_type,
            is_malloc_array_type, is_list, m_dims, n_dims, a_kind, module.get());
        if( is_integer ) {
            int64_t min_value = INT64_MAX;
            int64_t max_value = INT64_MIN;
            size_t max_name_len = 0;
            llvm::Value* itr_value = nullptr;
            for( auto itr: x.m_symtab->get_scope() ) {
                ASR::Variable_t* itr_var = ASR::down_cast<ASR::Variable_t>(itr.second);
                ASR::expr_t* value = ASRUtils::expr_value(itr_var->m_symbolic_value);
                int64_t value_int64 = -1;
                this->visit_expr(*value);
                itr_value = tmp;
                ASRUtils::extract_value(value, value_int64);
                min_value = std::min(value_int64, min_value);
                max_value = std::max(value_int64, max_value);
                max_name_len = std::max(max_name_len, itr.first.size());
            }

            llvm::ArrayType* name_array_type = llvm::ArrayType::get(llvm::Type::getInt8Ty(context),
                                                                    max_name_len + 1);
            llvm::StructType* enum_value_type = llvm::StructType::create({name_array_type, value_type});
            llvm::Constant* empty_vt = llvm::ConstantStruct::get(enum_value_type, {llvm::ConstantArray::get(name_array_type,
                {llvm::ConstantInt::get(llvm::Type::getInt8Ty(context), llvm::APInt(8, '\0'))}),
                (llvm::Constant*) itr_value});
            std::vector<llvm::Constant*> enum_value_pairs(max_value - min_value + 1, empty_vt);

            for( auto itr: x.m_symtab->get_scope() ) {
                ASR::Variable_t* itr_var = ASR::down_cast<ASR::Variable_t>(itr.second);
                ASR::expr_t* value = ASRUtils::expr_value(itr_var->m_symbolic_value);
                int64_t value_int64 = -1;
                ASRUtils::extract_value(value, value_int64);
                this->visit_expr(*value);
                std::vector<llvm::Constant*> itr_var_name_v;
                itr_var_name_v.reserve(itr.first.size());
                for( size_t i = 0; i < itr.first.size(); i++ ) {
                    itr_var_name_v.push_back(llvm::ConstantInt::get(
                        llvm::Type::getInt8Ty(context), llvm::APInt(8, itr_var->m_name[i])));
                }
                itr_var_name_v.push_back(llvm::ConstantInt::get(
                    llvm::Type::getInt8Ty(context), llvm::APInt(8, '\0')));
                llvm::Constant* name = llvm::ConstantArray::get(name_array_type, itr_var_name_v);
                enum_value_pairs[value_int64 - min_value] = llvm::ConstantStruct::get(
                    enum_value_type, {name, (llvm::Constant*) tmp});
            }

            llvm::ArrayType* global_enum_array = llvm::ArrayType::get(enum_value_type,
                                                    max_value - min_value + 1);
            llvm::Constant *array = module->getOrInsertGlobal(x.m_name,
                                        global_enum_array);
            module->getNamedGlobal(x.m_name)->setInitializer(
                llvm::ConstantArray::get(global_enum_array, enum_value_pairs));
            uint32_t h = get_hash((ASR::asr_t*)&x);
            llvm_symtab[h] = array;
        } else {
            size_t max_name_len = 0;

            for( auto itr: x.m_symtab->get_scope() ) {
                max_name_len = std::max(max_name_len, itr.first.size());
            }

            llvm::ArrayType* name_array_type = llvm::ArrayType::get(llvm::Type::getInt8Ty(context),
                                                                    max_name_len + 1);
            llvm::StructType* enum_value_type = llvm::StructType::create({name_array_type, value_type});
            std::vector<llvm::Constant*> enum_value_pairs(x.n_members, nullptr);

            for( auto itr: x.m_symtab->get_scope() ) {
                ASR::Variable_t* itr_var = ASR::down_cast<ASR::Variable_t>(itr.second);
                ASR::expr_t* value = itr_var->m_symbolic_value;
                int64_t value_int64 = -1;
                ASRUtils::extract_value(value, value_int64);
                this->visit_expr(*value);
                std::vector<llvm::Constant*> itr_var_name_v;
                itr_var_name_v.reserve(itr.first.size());
                for( size_t i = 0; i < itr.first.size(); i++ ) {
                    itr_var_name_v.push_back(llvm::ConstantInt::get(
                        llvm::Type::getInt8Ty(context), llvm::APInt(8, itr_var->m_name[i])));
                }
                itr_var_name_v.push_back(llvm::ConstantInt::get(
                    llvm::Type::getInt8Ty(context), llvm::APInt(8, '\0')));
                llvm::Constant* name = llvm::ConstantArray::get(name_array_type, itr_var_name_v);
                size_t dest_idx = 0;
                for( size_t j = 0; j < x.n_members; j++ ) {
                    if( std::string(x.m_members[j]) == itr.first ) {
                        dest_idx = j;
                        break ;
                    }
                }
                enum_value_pairs[dest_idx] = llvm::ConstantStruct::get(
                    enum_value_type, {name, (llvm::Constant*) tmp});
            }

            llvm::ArrayType* global_enum_array = llvm::ArrayType::get(enum_value_type,
                                                    x.n_members);
            llvm::Constant *array = module->getOrInsertGlobal(x.m_name,
                                        global_enum_array);
            module->getNamedGlobal(x.m_name)->setInitializer(
                llvm::ConstantArray::get(global_enum_array, enum_value_pairs));
            uint32_t h = get_hash((ASR::asr_t*)&x);
            llvm_symtab[h] = array;
        }
    }

    void start_module_init_function_prototype(const ASR::Module_t &x) {
        uint32_t h = get_hash((ASR::asr_t*)&x);
        llvm::FunctionType *function_type = llvm::FunctionType::get(
                llvm::Type::getVoidTy(context), {}, false);
        LCOMPILERS_ASSERT(llvm_symtab_fn.find(h) == llvm_symtab_fn.end());
        std::string module_fn_name = "__lfortran_module_init_" + std::string(x.m_name);
        llvm::Function *F = llvm::Function::Create(function_type,
                llvm::Function::ExternalLinkage, module_fn_name, module.get());
        llvm::BasicBlock *BB = llvm::BasicBlock::Create(context, ".entry", F);
        builder->SetInsertPoint(BB);

        llvm_symtab_fn[h] = F;
    }

    void finish_module_init_function_prototype(const ASR::Module_t &x) {
        uint32_t h = get_hash((ASR::asr_t*)&x);
        builder->CreateRetVoid();
        llvm_symtab_fn[h]->removeFromParent();
    }

    void visit_Module(const ASR::Module_t &x) {
        SymbolTable* current_scope_copy = current_scope;
        current_scope = x.m_symtab;
        mangle_prefix = "__module_" + std::string(x.m_name) + "_";

        start_module_init_function_prototype(x);

        for (auto &item : x.m_symtab->get_scope()) {
            if (is_a<ASR::Variable_t>(*item.second)) {
                ASR::Variable_t *v = down_cast<ASR::Variable_t>(
                        item.second);
                if( v->m_storage != ASR::storage_typeType::Parameter ) {
                    visit_Variable(*v);
                }
            } else if (is_a<ASR::Function_t>(*item.second)) {
                ASR::Function_t *v = down_cast<ASR::Function_t>(
                        item.second);
                ASR::FunctionType_t* func_type = ASR::down_cast<ASR::FunctionType_t>(v->m_function_signature);
                if (x.m_parent_module && func_type->m_module) {
                    ASR::Module_t* parent_module = ASR::down_cast<ASR::Module_t>(x.m_parent_module);
                    mangle_prefix = "__module_" + std::string(parent_module->m_name) + "_";
                }
                instantiate_function(*v);
                mangle_prefix = "__module_" + std::string(x.m_name) + "_";
            }
        }
        finish_module_init_function_prototype(x);

        visit_procedures(x);
        mangle_prefix = "";
        current_scope = current_scope_copy;
    }

#ifdef HAVE_TARGET_WASM
    void add_wasm_start_function() {
        llvm::FunctionType *function_type = llvm::FunctionType::get(
            llvm::Type::getVoidTy(context), {}, false);
        llvm::Function *F = llvm::Function::Create(function_type,
                llvm::Function::ExternalLinkage, "_start", module.get());
        llvm::BasicBlock *BB = llvm::BasicBlock::Create(context, ".entry", F);
        builder->SetInsertPoint(BB);
        std::vector<llvm::Value *> args;
        args.push_back(llvm::ConstantInt::get(context, llvm::APInt(32, 0)));
        args.push_back(llvm_utils->CreateAlloca(*builder, character_type));
        builder->CreateCall(module->getFunction("main"), args);
        builder->CreateRet(nullptr);
    }
#endif

    void visit_Program(const ASR::Program_t &x) {
        loop_head.clear();
        loop_head_names.clear();
        loop_or_block_end.clear();
        loop_or_block_end_names.clear();
        heap_arrays.clear();
        strings_to_be_deallocated.reserve(al, 1);
        SymbolTable* current_scope_copy = current_scope;
        current_scope = x.m_symtab;
        bool is_dict_present_copy_lp = dict_api_lp->is_dict_present();
        bool is_dict_present_copy_sc = dict_api_sc->is_dict_present();
        dict_api_lp->set_is_dict_present(false);
        dict_api_sc->set_is_dict_present(false);
        bool is_set_present_copy_lp = set_api_lp->is_set_present();
        bool is_set_present_copy_sc = set_api_sc->is_set_present();
        set_api_lp->set_is_set_present(false);
        set_api_sc->set_is_set_present(false);
        llvm_goto_targets.clear();
        // Generate code for the main program
        std::vector<llvm::Type*> command_line_args = {
            llvm::Type::getInt32Ty(context),
            character_type->getPointerTo()
        };
        llvm::FunctionType *function_type = llvm::FunctionType::get(
                llvm::Type::getInt32Ty(context), command_line_args, false);
        llvm::Function *F = llvm::Function::Create(function_type,
                llvm::Function::ExternalLinkage, "main", module.get());
        llvm::BasicBlock *BB = llvm::BasicBlock::Create(context,
                ".entry", F);
        if (compiler_options.emit_debug_info) {
            llvm::DISubprogram *SP;
            debug_emit_function(x, SP);
            F->setSubprogram(SP);
        }
        builder->SetInsertPoint(BB);
        // there maybe a possibility that nested function has an array variable
        // whose dimension depends on variable present in this program / function
        // thereby visit all integer variables and declare those:
        for(auto &item: x.m_symtab->get_scope()) {
            ASR::symbol_t* sym = item.second;
            if ( is_a<ASR::Variable_t>(*sym) ) {
                ASR::Variable_t* v = down_cast<ASR::Variable_t>(sym);
                uint32_t debug_arg_count = 0;
                if ( ASR::is_a<ASR::Integer_t>(*v->m_type) ) {
                    process_Variable(sym, x, debug_arg_count);
                }
            }
        }

        // Generate code for nested subroutines and functions first:
        for (auto &item : x.m_symtab->get_scope()) {
            if (is_a<ASR::Function_t>(*item.second)) {
                ASR::Function_t *v = down_cast<ASR::Function_t>(
                        item.second);
                instantiate_function(*v);
            }
        }
        visit_procedures(x);

        builder->SetInsertPoint(BB);
        // Call the `_lpython_call_initial_functions` function to assign command line argument
        // values to `argc` and `argv`, and set the random seed to the system clock.
        {
            if (compiler_options.emit_debug_info) debug_emit_loc(x);
            llvm::Function *fn = module->getFunction("_lpython_call_initial_functions");
            if(!fn) {
                llvm::FunctionType *function_type = llvm::FunctionType::get(
                    llvm::Type::getVoidTy(context), {
                        llvm::Type::getInt32Ty(context),
                        character_type->getPointerTo()
                    }, false);
                fn = llvm::Function::Create(function_type,
                    llvm::Function::ExternalLinkage, "_lpython_call_initial_functions", *module);
            }
            std::vector<llvm::Value *> args;
            for (llvm::Argument &llvm_arg : F->args()) {
                args.push_back(&llvm_arg);
            }
            builder->CreateCall(fn, args);
        }
        for(to_be_allocated_array array : allocatable_array_details){
                    fill_array_details_(array.expr, array.pointer_to_array_type, array.array_type, nullptr, array.n_dims,
                true, true, false, array.var_type);
        }
        declare_vars(x);
        for(variable_inital_value var_to_initalize : variable_inital_value_vec){
            set_VariableInital_value(var_to_initalize.v, var_to_initalize.target_var);
        }
        proc_return = llvm::BasicBlock::Create(context, "return");
        for (size_t i=0; i<x.n_body; i++) {
            this->visit_stmt(*x.m_body[i]);
        }
        for( auto& value: heap_arrays ) {
            LLVM::lfortran_free(context, *module, *builder, value);
        }

        {
            llvm::Function *fn = module->getFunction("_lpython_free_argv");
            if(!fn) {
                llvm::FunctionType *function_type = llvm::FunctionType::get(
                    llvm::Type::getVoidTy(context), {}, false);
                fn = llvm::Function::Create(function_type,
                    llvm::Function::ExternalLinkage, "_lpython_free_argv", *module);
            }
            builder->CreateCall(fn, {});
        }

        start_new_block(proc_return);
        llvm::Value *ret_val2 = llvm::ConstantInt::get(context,
            llvm::APInt(32, 0));
        builder->CreateRet(ret_val2);
        dict_api_lp->set_is_dict_present(is_dict_present_copy_lp);
        dict_api_sc->set_is_dict_present(is_dict_present_copy_sc);
        set_api_lp->set_is_set_present(is_set_present_copy_lp);
        set_api_sc->set_is_set_present(is_set_present_copy_sc);

        // Finalize the debug info.
        if (compiler_options.emit_debug_info) DBuilder->finalize();
        current_scope = current_scope_copy;
        loop_head.clear();
        loop_head_names.clear();
        loop_or_block_end.clear();
        loop_or_block_end_names.clear();
        heap_arrays.clear();
        strings_to_be_deallocated.reserve(al, 1);

#ifdef HAVE_TARGET_WASM
        if (startswith(compiler_options.target, "wasm")) {
            add_wasm_start_function();
        }
#endif
    }

    /*
    * This function detects if the current variable is an argument.
    * of a function or argument. Some manipulations are to be done
    * only on arguments and not on local variables.
    */
    bool is_argument(ASR::Variable_t* v, ASR::expr_t** m_args,
                        int n_args) {
        for( int i = 0; i < n_args; i++ ) {
            ASR::expr_t* m_arg = m_args[i];
            uint32_t h_m_arg = get_hash((ASR::asr_t*)m_arg);
            uint32_t h_v = get_hash((ASR::asr_t*)v);
            if( h_m_arg == h_v ) {
                return true;
            }
        }
        return false;
    }

    void fill_array_details_(ASR::expr_t* expr, llvm::Value* ptr, llvm::Type* type_, ASR::dimension_t* m_dims,
        size_t n_dims, bool is_malloc_array_type, bool is_array_type,
        bool is_list, [[maybe_unused]]ASR::ttype_t* m_type, bool is_data_only=false) {
        ASR::ttype_t* asr_data_type = ASRUtils::type_get_past_array(
            ASRUtils::type_get_past_pointer(
                ASRUtils::type_get_past_allocatable(ASRUtils::expr_type(expr))));
        llvm::Type* llvm_data_type = llvm_utils->get_type_from_ttype_t_util(expr, asr_data_type, module.get());
        llvm::Value* ptr_ = nullptr;
        if( is_malloc_array_type && !is_list && !is_data_only ) {
            ptr_ = llvm_utils->CreateAlloca(*builder, type_, nullptr, "arr_desc");
            if(ASRUtils::is_character(*m_type)){
                llvm::Value* str_desc = create_and_setup_string_for_array(m_type, nullptr, false, "arr_desc_str_desc");
                builder->CreateStore(str_desc, arr_descr->get_pointer_to_data(ptr_));
            }
            arr_descr->fill_dimension_descriptor(ptr_, n_dims);
        }
        if( is_array_type && !is_malloc_array_type &&
            !is_list ) {
            fill_array_details(ptr, llvm_data_type, m_dims, n_dims, is_data_only);
        }
        if( is_array_type && is_malloc_array_type &&
            !is_list && !is_data_only && 
            !ASRUtils::is_character(*m_type) /*already nullified*/) {
            // Set allocatable arrays as unallocated
            LCOMPILERS_ASSERT(ptr_ != nullptr);
            arr_descr->reset_is_allocated_flag(ptr_, llvm_data_type);
        }
        if( ptr_ ) {
            LLVM::CreateStore(*builder, ptr_, ptr);
        }
    }

    void set_pointer_variable_to_null(ASR::Variable_t* v, llvm::Value* null_value, llvm::Value* ptr) {
        if( (ASR::is_a<ASR::Allocatable_t>(*v->m_type) ||
                ASR::is_a<ASR::Pointer_t>(*v->m_type)) && 
            (v->m_intent == ASRUtils::intent_local || 
             v->m_intent == ASRUtils::intent_return_var ) &&
             !ASRUtils::is_character(*v->m_type)) {
            if (ASRUtils::is_class_type(ASRUtils::type_get_past_allocatable_pointer(v->m_type))) { 
                ASR::symbol_t* struct_sym = ASRUtils::symbol_get_past_external(v->m_type_declaration); 
                ASR::Struct_t* st = ASR::down_cast<ASR::Struct_t>(struct_sym); 
                llvm::Type* wrapper_struct_llvm_type = get_llvm_struct_data_type(st, false); 
                llvm::Value* struct_hash = llvm::ConstantInt::get(llvm_utils->getIntType(8), 
                                        llvm::APInt(64, get_class_hash(struct_sym))); 
                llvm::Type* v_llvm_type = llvm_utils->get_type_from_ttype_t_util(ASRUtils::EXPR(ASR::make_Var_t( 
                    al, v->base.base.loc, &v->base)), v->m_type, module.get());
                llvm::Value* hash_ptr = llvm_utils->create_gep2(v_llvm_type, ptr, 0); 
                builder->CreateStore(struct_hash, hash_ptr); 
                llvm::Value* struct_ptr = llvm_utils->create_gep2(v_llvm_type, ptr, 1); 
                builder->CreateStore(llvm::ConstantPointerNull::getNullValue(wrapper_struct_llvm_type->getPointerTo()), struct_ptr); 
            } else { 
                builder->CreateStore(null_value, ptr); 
            }
        } 
    }


    void allocate_array_members_of_struct(ASR::Struct_t* struct_sym, llvm::Value* ptr, ASR::ttype_t* asr_type, bool is_intent_out = false) {
        LCOMPILERS_ASSERT(ASR::is_a<ASR::StructType_t>(*asr_type));
        ASR::Struct_t* struct_type_t = nullptr;
        if (ASR::is_a<ASR::StructType_t>(*asr_type)) {
            struct_type_t = struct_sym;
        }
        while (struct_type_t) {
            std::string struct_type_name = struct_type_t->m_name;
            for( auto item: struct_type_t->m_symtab->get_scope() ) {
                ASR::symbol_t *sym = ASRUtils::symbol_get_past_external(item.second);
                if (name2memidx[struct_type_name].find(item.first) == name2memidx[struct_type_name].end()) {
                    continue;
                }
                if( ASR::is_a<ASR::StructMethodDeclaration_t>(*sym) ||
                    ASR::is_a<ASR::GenericProcedure_t>(*sym) ||
                    ASR::is_a<ASR::Union_t>(*sym) ||
                    ASR::is_a<ASR::Struct_t>(*sym) ||
                    ASR::is_a<ASR::CustomOperator_t>(*sym) ) {
                    continue ;
                }
                ASR::ttype_t* symbol_type = ASRUtils::symbol_type(sym);
                int idx = name2memidx[struct_type_name][item.first];
                llvm::Type* type = name2dertype[struct_type_name];
                llvm::Value* ptr_member = llvm_utils->create_gep2(type, ptr, idx);
                ASR::Variable_t* v = nullptr;
                // Don't reinitialize arrays where the type's intent is out because array descriptor is allocated on the stack
                // and might be returned.
                if( ASR::is_a<ASR::Variable_t>(*sym) && !(is_intent_out ) ) {
                    v = ASR::down_cast<ASR::Variable_t>(sym);
                    set_pointer_variable_to_null(v, llvm::Constant::getNullValue(
                        llvm_utils->get_type_from_ttype_t_util(ASRUtils::EXPR(ASR::make_Var_t(
                    al, v->base.base.loc, &v->base)), v->m_type, module.get())),
                        ptr_member);
                }
                if( ASRUtils::is_array(symbol_type) && v && !is_intent_out) {
                    ASR::dimension_t* m_dims = nullptr;
                    size_t n_dims = ASRUtils::extract_dimensions_from_ttype(symbol_type, m_dims);
                    ASR::array_physical_typeType phy_type = ASRUtils::extract_physical_type(symbol_type);
                    bool is_data_only = (phy_type == ASR::array_physical_typeType::PointerToDataArray ||
                                        phy_type == ASR::array_physical_typeType::FixedSizeArray ||
                                        (phy_type == ASR::array_physical_typeType::StringArraySinglePointer &&
                                        ASRUtils::is_fixed_size_array(symbol_type)));
                    if (phy_type == ASR::array_physical_typeType::DescriptorArray ||
                        (phy_type == ASR::array_physical_typeType::StringArraySinglePointer &&
                        ASRUtils::is_dimension_empty(m_dims, n_dims))) {
                        int n_dims = 0, a_kind=4;
                        ASR::dimension_t* m_dims = nullptr;
                        bool is_array_type = false;
                        bool is_malloc_array_type = false;
                        bool is_list = false;
                        llvm_utils->get_type_from_ttype_t(ASRUtils::EXPR(ASR::make_Var_t(
                    al, v->base.base.loc, &v->base)), v->m_type,
                                    v->m_type_declaration, v->m_storage, is_array_type,
                                    is_malloc_array_type, is_list, m_dims, n_dims, a_kind,
                                    module.get());
                        llvm::Type* type_ = llvm_utils->get_type_from_ttype_t_util(ASRUtils::EXPR(
                ASR::make_Var_t(al, v->base.base.loc, &v->base)), 
                            ASRUtils::type_get_past_pointer(
                                ASRUtils::type_get_past_allocatable(v->m_type)), module.get(), v->m_abi);
                        fill_array_details_(ASRUtils::EXPR(ASR::make_Var_t(
                    al, v->base.base.loc, &v->base)), ptr_member, type_, m_dims, n_dims,
                                is_malloc_array_type, is_array_type, is_list, v->m_type);
                    } else {
                        fill_array_details_(ASRUtils::EXPR(ASR::make_Var_t(
                    al, v->base.base.loc, &v->base)), ptr_member, nullptr, m_dims, n_dims, false, true, false, symbol_type, is_data_only);
                    }
                    if (ASR::is_a<ASR::StructType_t>(*ASRUtils::type_get_past_array(symbol_type))
                        && !ASRUtils::is_class_type(ASRUtils::type_get_past_array(symbol_type))) {
#if LLVM_VERSION_MAJOR > 16
                        ptr_type[ptr_member] = llvm_utils->get_type_from_ttype_t_util(ASRUtils::get_expr_from_sym(al, sym),
                            symbol_type, module.get());
#endif
                        allocate_array_members_of_struct_arrays(ASRUtils::get_expr_from_sym(al, sym), ptr_member, symbol_type);
                    }
                } else if (ASR::is_a<ASR::StructType_t>(*symbol_type) && !ASRUtils::is_class_type(symbol_type)) {
                    ASR::Struct_t* struct_sym = ASR::down_cast<ASR::Struct_t>(ASRUtils::symbol_get_past_external(
                        ASR::down_cast<ASR::Variable_t>(sym)->m_type_declaration));
                    allocate_array_members_of_struct(struct_sym, ptr_member, symbol_type);
                }  else if(ASRUtils::is_string_only(symbol_type) && !is_intent_out) {
                    setup_string(ptr_member, symbol_type);
                }
                if( ASR::is_a<ASR::Variable_t>(*sym) ) {
                    v = ASR::down_cast<ASR::Variable_t>(sym);
                    if( v->m_symbolic_value ) {
                        visit_expr(*v->m_symbolic_value);
                        if( ASR::is_a<ASR::PointerNullConstant_t>(*v->m_symbolic_value) &&
                            ASRUtils::is_array(v->m_type)){ // Store into array's data pointer.
                            LCOMPILERS_ASSERT(ASR::is_a<ASR::Pointer_t>(*v->m_type));
                            llvm::Value* pointer_array = builder->CreateLoad(
                                llvm_utils->get_type_from_ttype_t_util(ASRUtils::EXPR(ASR::make_Var_t(
                                al, v->base.base.loc, &v->base)), v->m_type, module.get()),
                                ptr_member);
                            llvm::Type* const array_desc_type = llvm_utils->arr_api->get_array_type(
                                ASRUtils::EXPR(ASR::make_Var_t(al, v->base.base.loc, (ASR::symbol_t*)v)),
                                ASRUtils::type_get_past_allocatable_pointer(v->m_type),
                                llvm_utils->get_el_type(
                                    ASRUtils::EXPR(ASR::make_Var_t(al, v->base.base.loc, &v->base)),
                                    ASRUtils::extract_type(v->m_type),
                                    module.get()),
                                false);
                            builder->CreateStore(
                                tmp, llvm_utils->create_gep2(array_desc_type, pointer_array, 0));
                        } else if(ASRUtils::is_string_only(expr_type(v->m_symbolic_value))) {
                            llvm_utils->lfortran_str_copy(
                            ptr_member, tmp,
                            ASRUtils::get_string_type(symbol_type),
                            ASRUtils::get_string_type(expr_type(v->m_symbolic_value)),
                            ASRUtils::is_allocatable(symbol_type));
                        } else if (ASRUtils::is_array(v->m_type)) {
                            ASR::ArrayConstant_t* arr_const = ASR::down_cast<ASR::ArrayConstant_t>(ASRUtils::expr_value(v->m_symbolic_value));
                            llvm::Type* array_type = llvm_utils->get_type_from_ttype_t_util(ASRUtils::expr_value(v->m_symbolic_value), arr_const->m_type, module.get());
                            llvm::Value* arg_size = llvm::ConstantInt::get(llvm::Type::getInt32Ty(context),
                            llvm::APInt(32, ASRUtils::get_fixed_size_of_array(arr_const->m_type)));
                            llvm::Type* llvm_data_type = llvm_utils->get_type_from_ttype_t_util(ASRUtils::expr_value(v->m_symbolic_value),
                                ASRUtils::type_get_past_array(ASRUtils::expr_type(v->m_symbolic_value)), module.get());
                            llvm::DataLayout data_layout(module->getDataLayout());
                            size_t dt_size = data_layout.getTypeAllocSize(llvm_data_type);
                            arg_size = builder->CreateMul(llvm::ConstantInt::get(
                                llvm::Type::getInt32Ty(context), llvm::APInt(32, dt_size)), arg_size);
                            builder->CreateMemCpy(llvm_utils->create_gep2(array_type,ptr_member, 0),
                                llvm::MaybeAlign(), tmp, llvm::MaybeAlign(), arg_size, v->m_is_volatile);
                        } else if ((ASRUtils::is_pointer(v->m_type) &&
                                !ASR::is_a<ASR::PointerNullConstant_t>(
                                    *v->m_symbolic_value))        ||
                            ASRUtils::is_allocatable(v->m_type)) { // Any non primitve
                            throw LCompilersException("Not implemented");
                        } else {
                            LLVM::CreateStore(*builder, tmp, ptr_member);
                        }
                    }
                }
            }
            if (struct_type_t->m_parent) {
                ptr = llvm_utils->create_gep2(name2dertype[struct_type_t->m_name], ptr, 0);
                struct_type_t = ASR::down_cast<ASR::Struct_t>(ASRUtils::symbol_get_past_external(struct_type_t->m_parent));
            } else {
                struct_type_t = nullptr;
            }
        }
    }

    void allocate_array_members_of_struct_arrays(ASR::expr_t* expr, llvm::Value* ptr, ASR::ttype_t* v_m_type) {
        ASR::array_physical_typeType phy_type = ASRUtils::extract_physical_type(v_m_type);
        llvm::Type* el_type = llvm_utils->get_type_from_ttype_t_util(expr,
            ASRUtils::extract_type(v_m_type), module.get());
        llvm::Value* array_size = llvm_utils->CreateAlloca(
                llvm::Type::getInt32Ty(context), nullptr, "array_size");
        switch( phy_type ) {
            case ASR::array_physical_typeType::FixedSizeArray: {
                ASR::dimension_t* m_dims = nullptr;
                size_t n_dims = ASRUtils::extract_dimensions_from_ttype(v_m_type, m_dims);
                LLVM::CreateStore(*builder, llvm::ConstantInt::get(llvm::Type::getInt32Ty(context),
                    llvm::APInt(32, ASRUtils::get_fixed_size_of_array(m_dims, n_dims))), array_size);
                break;
            }
            case ASR::array_physical_typeType::DescriptorArray: {
                llvm::Value* array_size_value = arr_descr->get_array_size(ptr, nullptr, 4);
                LLVM::CreateStore(*builder, array_size_value, array_size);
                break;
            }
            case ASR::array_physical_typeType::PointerToDataArray: {
                ASR::dimension_t* m_dims = nullptr;
                size_t n_dims = ASRUtils::extract_dimensions_from_ttype(v_m_type, m_dims);
                llvm::Value* llvm_size = llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), llvm::APInt(32, 1));
                int ptr_loads_copy = ptr_loads;
                ptr_loads = 2;
                for( size_t i = 0; i < n_dims; i++ ) {
                    visit_expr_wrapper(m_dims[i].m_length, true);
                    llvm_size = builder->CreateMul(tmp, llvm_size);
                }
                ptr_loads = ptr_loads_copy;
                LLVM::CreateStore(*builder, llvm_size, array_size);
                break;
            }
            default: {
                LCOMPILERS_ASSERT_MSG(false, std::to_string(phy_type));
            }
        }
        llvm::Value* llvmi = llvm_utils->CreateAlloca(llvm::Type::getInt32Ty(context), nullptr, "i");
        LLVM::CreateStore(*builder,
            llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), llvm::APInt(32, 0)), llvmi);
        create_loop(nullptr, [=]() {
                llvm::Value* llvmi_loaded = llvm_utils->CreateLoad(llvmi);
                llvm::Value* array_size_loaded = llvm_utils->CreateLoad(array_size);
                return builder->CreateICmpSLT(
                    llvmi_loaded, array_size_loaded);
            },
            [=]() {
                llvm::Value* ptr_i = nullptr;
                switch (phy_type) {
                    case ASR::array_physical_typeType::FixedSizeArray: {
                        llvm::Type* ptr_i_type = llvm_utils->get_type_from_ttype_t_util(expr, v_m_type, module.get());
                        ptr_i = llvm_utils->create_gep2(ptr_i_type, ptr, llvm_utils->CreateLoad(llvmi));
                        break;
                    }
                    case ASR::array_physical_typeType::DescriptorArray: {
                        ptr_i = llvm_utils->create_ptr_gep2(el_type,
                            llvm_utils->CreateLoad2(el_type->getPointerTo(), arr_descr->get_pointer_to_data(ptr)),
                            llvm_utils->CreateLoad(llvmi));
                        break;
                    }
                    case ASR::array_physical_typeType::PointerToDataArray: {
#if LLVM_VERSION_MAJOR > 16
                        ptr_type[ptr] = llvm_utils->get_type_from_ttype_t_util(expr, v_m_type, module.get());
#endif
                        ptr_i = llvm_utils->create_ptr_gep(ptr, llvm_utils->CreateLoad(llvmi));
                        break;
                    }
                    default: {
                        LCOMPILERS_ASSERT(false);
                    }
                }
                allocate_array_members_of_struct(
                    ASR::down_cast<ASR::Struct_t>(ASRUtils::symbol_get_past_external(ASRUtils::get_struct_sym_from_struct_expr(expr))),
                        ptr_i, ASRUtils::extract_type(v_m_type));
                LLVM::CreateStore(*builder,
                    builder->CreateAdd(llvm_utils->CreateLoad(llvmi),
                        llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), llvm::APInt(32, 1))),
                    llvmi);
            });
    }

    void create_vtab_for_struct_type(ASR::symbol_t* struct_type_sym, SymbolTable* symtab) {
        LCOMPILERS_ASSERT(ASR::is_a<ASR::Struct_t>(*struct_type_sym));
        ASR::Struct_t* struct_type_t = ASR::down_cast<ASR::Struct_t>(struct_type_sym);
        if( type2vtab.find(struct_type_sym) != type2vtab.end() &&
            type2vtab[struct_type_sym].find(symtab) != type2vtab[struct_type_sym].end() ) {
            return ;
        }
        if( type2vtabtype.find(struct_type_sym) == type2vtabtype.end() ) {
            std::vector<llvm::Type*> type_vec = {llvm_utils->getIntType(8)};
            type2vtabtype[struct_type_sym] = llvm::StructType::create(
                                                context, type_vec,
                                                std::string("__vtab_") +
                                                std::string(struct_type_t->m_name));
        }
        llvm::Type* vtab_type = type2vtabtype[struct_type_sym];
        llvm::Value* vtab_obj = llvm_utils->CreateAlloca(*builder, vtab_type);
        llvm::Value* struct_type_hash_ptr = llvm_utils->create_gep2(vtab_type, vtab_obj, 0);
        llvm::Value* struct_type_hash = llvm::ConstantInt::get(llvm_utils->getIntType(8),
            llvm::APInt(64, get_class_hash(struct_type_sym)));
        builder->CreateStore(struct_type_hash, struct_type_hash_ptr);
        type2vtab[struct_type_sym][symtab] = vtab_obj;
        ASR::symbol_t* struct_type_ = struct_type_sym;
        bool base_found = false;
        while( !base_found ) {
            if( ASR::is_a<ASR::Struct_t>(*struct_type_) ) {
                ASR::Struct_t* struct_type = ASR::down_cast<ASR::Struct_t>(struct_type_);
                if( struct_type->m_parent == nullptr ) {
                    base_found = true;
                } else {
                    struct_type_ = ASRUtils::symbol_get_past_external(struct_type->m_parent);
                }
            } else {
                LCOMPILERS_ASSERT(false);
            }
        }
        class2vtab[struct_type_][symtab].push_back(vtab_obj);
    }

    void collect_variable_types_and_struct_types(
        std::set<std::string>& variable_type_names,
        std::vector<ASR::symbol_t*>& struct_types,
        SymbolTable* x_symtab) {
        if (x_symtab == nullptr) {
            return ;
        }
        for (auto &item : x_symtab->get_scope()) {
            ASR::symbol_t* var_sym = item.second;
            if (ASR::is_a<ASR::Variable_t>(*var_sym)) {
                ASR::Variable_t *v = ASR::down_cast<ASR::Variable_t>(var_sym);
                ASR::ttype_t* v_type = ASRUtils::extract_type(v->m_type);
                if (ASR::is_a<ASR::StructType_t>(*v_type)) {
                    variable_type_names.insert(ASRUtils::symbol_name(var_sym));
                }
            } else if (ASR::is_a<ASR::Struct_t>(
                        *ASRUtils::symbol_get_past_external(var_sym))) {
                struct_types.push_back(ASRUtils::symbol_get_past_external(var_sym));
            }
        }
        collect_variable_types_and_struct_types(variable_type_names, struct_types, x_symtab->parent);
    }
    void set_VariableInital_value(ASR::Variable_t* v, llvm::Value* target_var){
        if (v->m_value != nullptr) {
            this->visit_expr_wrapper(v->m_value, true, v->m_is_volatile);
        } else {
            this->visit_expr_wrapper(v->m_symbolic_value, true, v->m_is_volatile);
        }
        llvm::Value *init_value = tmp;
        if( ASRUtils::is_array(v->m_type) &&
            ASRUtils::is_array(ASRUtils::expr_type(v->m_symbolic_value)) &&
            (ASR::is_a<ASR::ArrayConstant_t>(*v->m_symbolic_value) ||
            (v->m_value && ASR::is_a<ASR::ArrayConstant_t>(*v->m_value)))) {
            ASR::array_physical_typeType target_ptype = ASRUtils::extract_physical_type(v->m_type);
            if( target_ptype == ASR::array_physical_typeType::DescriptorArray ) {
                target_var = arr_descr->get_pointer_to_data(target_var);
                builder->CreateStore(init_value, target_var, v->m_is_volatile);
            } else if( target_ptype == ASR::array_physical_typeType::FixedSizeArray ) {
                llvm::Value* arg_size = llvm::ConstantInt::get(llvm::Type::getInt32Ty(context),
                llvm::APInt(32, ASRUtils::get_fixed_size_of_array(ASR::down_cast<ASR::ArrayConstant_t>(v->m_value)->m_type)));
                llvm::Type* llvm_data_type = llvm_utils->get_type_from_ttype_t_util(ASRUtils::EXPR(ASR::make_Var_t(
                    al, v->base.base.loc, &v->base)),
                    ASRUtils::type_get_past_array(ASRUtils::expr_type(v->m_value)), module.get());
                llvm::DataLayout data_layout(module->getDataLayout());
                size_t dt_size = data_layout.getTypeAllocSize(llvm_data_type);
                arg_size = builder->CreateMul(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(context), llvm::APInt(32, dt_size)), arg_size);
                llvm::Type* llvm_type
                    = llvm_utils->get_type_from_ttype_t_util(ASRUtils::EXPR(ASR::make_Var_t(
                    al, v->base.base.loc, &v->base)), v->m_type, module.get());
                builder->CreateMemCpy(llvm_utils->create_gep2(llvm_type,target_var, 0),
                    llvm::MaybeAlign(), init_value, llvm::MaybeAlign(), arg_size, v->m_is_volatile);
            } else if(target_ptype == ASR::PointerToDataArray){
                if(ASRUtils::is_array_of_strings(v->m_type)){
                    builder->CreateMemCpy(
                        llvm_utils->get_stringArray_data(v->m_type, target_var), llvm::MaybeAlign(),
                        llvm_utils->get_stringArray_data(v->m_type, init_value), llvm::MaybeAlign(),
                        llvm_utils->get_stringArray_whole_size(v->m_type));
                } else {
                    throw LCompilersException("Unhandled Array Type");
                }
            } else {
                throw LCompilersException("Unhandled ArrayPhysicalType");
            }
        } else if (ASR::is_a<ASR::ArrayReshape_t>(*v->m_symbolic_value)) {
            builder->CreateStore(llvm_utils->CreateLoad(init_value), target_var, v->m_is_volatile);
        } else if (is_a<ASR::String_t>(*v->m_type)) {
            ASR::String_t *t = down_cast<ASR::String_t>(v->m_type);
            visit_expr(*t->m_len);
            if( t->m_len_kind == ASR::string_length_kindType::DeferredLength ){LCOMPILERS_ASSERT(false);}
            // target decides if the str_copy is performed on string descriptor or pointer.
            tmp = llvm_utils->lfortran_str_copy(
                target_var, init_value,
                ASR::down_cast<ASR::String_t>(ASRUtils::extract_type(v->m_type)),
                ASR::down_cast<ASR::String_t>((ASRUtils::extract_type(expr_type(v->m_symbolic_value? v->m_symbolic_value : v->m_value)))),
                ASRUtils::is_allocatable(v->m_type));
            if (v->m_intent == intent_local) {
                llvm::Type* v_llvm_type = llvm_utils->get_type_from_ttype_t_util(ASRUtils::EXPR(ASR::make_Var_t(
                    al, v->base.base.loc, &v->base)), v->m_type, module.get());
                strings_to_be_deallocated.push_back(al, llvm_utils->CreateLoad2(v_llvm_type, target_var, v->m_is_volatile));
            }
        } else if(ASRUtils::is_array(v->m_type) &&
                (ASR::is_a<ASR::PointerNullConstant_t>(*v->m_symbolic_value) ||
                ASR::is_a<ASR::PointerNullConstant_t>(*v->m_value))){
                LCOMPILERS_ASSERT(ASR::is_a<ASR::Pointer_t>(*v->m_type));
                LCOMPILERS_ASSERT(ASRUtils::extract_physical_type(v->m_type) ==
                                     ASR::array_physical_typeType::DescriptorArray);
                llvm::Type* const array_desc_type = llvm_utils->arr_api->get_array_type(
                    ASRUtils::EXPR(ASR::make_Var_t(al, v->base.base.loc, (ASR::symbol_t*)v)),
                    ASRUtils::type_get_past_allocatable_pointer(v->m_type),
                    llvm_utils->get_el_type(
                        ASRUtils::EXPR(ASR::make_Var_t(al, v->base.base.loc, &v->base)),
                        ASRUtils::extract_type(v->m_type),
                        module.get()),
                    false);
                llvm::Value* data_ptr = llvm_utils->create_gep2(
                    array_desc_type, llvm_utils->CreateLoad(target_var), 0);
                builder->CreateStore(init_value, data_ptr, v->m_is_volatile);
        } else {
            if (v->m_storage == ASR::storage_typeType::Save
                && v->m_value
                && (ASR::is_a<ASR::Integer_t>(*v->m_type)
                || ASR::is_a<ASR::Real_t>(*v->m_type)
                || ASR::is_a<ASR::Logical_t>(*v->m_type)) && !v->m_is_volatile) {
                // Do nothing, the value is already initialized
                // in the global variable
            } else {
                builder->CreateStore(init_value, target_var, v->m_is_volatile);
            }
        }
    }

    template<typename T>
    void process_Variable(ASR::symbol_t* var_sym, T& x, uint32_t &debug_arg_count) {
        llvm::Value *target_var = nullptr;
        ASR::Variable_t *v = down_cast<ASR::Variable_t>(var_sym);
        uint32_t h = get_hash((ASR::asr_t*)v);
        llvm::Type *type;
        int n_dims = 0, a_kind = 4;
        ASR::dimension_t* m_dims = nullptr;
        bool is_array_type = false;
        bool is_malloc_array_type = false;
        bool is_list = false;
        bool is_dict = ASR::is_a<ASR::Dict_t>(*v->m_type);
        if (v->m_intent == intent_local ||
            v->m_intent == intent_return_var ||
            !v->m_intent) {
            type = llvm_utils->get_type_from_ttype_t(ASRUtils::EXPR(ASR::make_Var_t(
                    al, v->base.base.loc, &v->base)),
                v->m_type, v->m_type_declaration, v->m_storage, is_array_type,
                is_malloc_array_type, is_list, m_dims, n_dims, a_kind, module.get());
            llvm::Type* type_ = llvm_utils->get_type_from_ttype_t_util(ASRUtils::EXPR(ASR::make_Var_t(
                    al, v->base.base.loc, &v->base)),
                ASRUtils::type_get_past_pointer(
                    ASRUtils::type_get_past_allocatable(v->m_type)), module.get(), v->m_abi);

            /*
            * The following if block is used for converting any
            * general array descriptor to a pointer type which
            * can be passed as an argument in a function call in LLVM IR.
            */
            if( x.class_type == ASR::symbolType::Function) {
                std::string m_name = std::string(x.m_name);
                ASR::Function_t* _func = (ASR::Function_t*)(&(x.base));
                std::uint32_t m_h = get_hash((ASR::asr_t*)_func);
                ASR::abiType abi_type = ASRUtils::get_FunctionType(_func)->m_abi;
                bool is_v_arg = is_argument(v, _func->m_args, _func->n_args);
                if( is_array_type && !is_list && !is_dict) {
                    /* The first element in an array descriptor can be either of
                    * llvm::ArrayType or llvm::PointerType. However, a
                    * function only accepts llvm::PointerType for arrays. Hence,
                    * the following if block extracts the pointer to first element
                    * of an array from its descriptor. Note that this happens only
                    * for arguments and not for local function variables.
                    */
                    if( abi_type == ASR::abiType::Source && is_v_arg ) {
                        type = arr_descr->get_argument_type(type, m_h, v->m_name, arr_arg_type_cache);
                        is_array_type = false;
                    } else if( abi_type == ASR::abiType::Intrinsic &&
                        fname2arg_type.find(m_name) != fname2arg_type.end() ) {
                        type = fname2arg_type[m_name].second;
                        is_array_type = false;
                    }
                }
            }

            llvm::Value* array_size = nullptr;
            bool is_array_of_strings = ASRUtils::is_array(v->m_type) && ASRUtils::is_character(*v->m_type);
            if( ASRUtils::is_array(v->m_type) &&
                ASRUtils::extract_physical_type(v->m_type) ==
                ASR::array_physical_typeType::PointerToDataArray &&
                !LLVM::is_llvm_pointer(*v->m_type) ) {
                type = llvm_utils->get_type_from_ttype_t(ASRUtils::EXPR(ASR::make_Var_t(
                    al, v->base.base.loc, &v->base)),
                        ASRUtils::type_get_past_array(v->m_type),
                        v->m_type_declaration, v->m_storage, is_array_type,
                        is_malloc_array_type, is_list, m_dims, n_dims, a_kind, module.get());
                ASR::dimension_t* m_dims = nullptr;
                size_t n_dims = ASRUtils::extract_dimensions_from_ttype(v->m_type, m_dims);
                array_size = llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), llvm::APInt(32, 1));
                int ptr_loads_copy = ptr_loads;
                ptr_loads = 2;
                for( size_t i = 0; i < n_dims; i++ ) {
                    this->visit_expr_wrapper(m_dims[i].m_length, true);

                    if (m_dims[i].m_length != nullptr &&  ASR::is_a<ASR::Var_t>(*m_dims[i].m_length)) {
                        ASR::Var_t* m_length_var = ASR::down_cast<ASR::Var_t>(m_dims[i].m_length);
                        ASR::symbol_t* m_length_sym = ASRUtils::symbol_get_past_external(m_length_var->m_v);
                        if (m_length_sym != nullptr && ASR::is_a<ASR::Variable_t>(*m_length_sym)) {
                            ASR::Variable_t* m_length_variable = ASR::down_cast<ASR::Variable_t>(m_length_sym);
                            uint32_t m_length_variable_h = get_hash((ASR::asr_t*)m_length_variable);
                            llvm::Type* deep_type = llvm_utils->get_type_from_ttype_t_util(m_dims[i].m_length, ASRUtils::expr_type(m_dims[i].m_length),module.get());
                            llvm::Value* deep = llvm_utils->CreateAlloca(*builder, deep_type, nullptr, "deep");
                            builder->CreateStore(tmp, deep, v->m_is_volatile);
                            llvm::Type* m_dims_length_llvm_type = llvm_utils->get_type_from_ttype_t_util(m_dims[i].m_length, ASRUtils::expr_type(m_dims[i].m_length), module.get());
                            tmp = llvm_utils->CreateLoad2(m_dims_length_llvm_type,deep, v->m_is_volatile);
                            llvm_symtab_deep_copy[m_length_variable_h] = deep;
                        }
                    }

                    // Make dimension length and return size compatible.(TODO : array_size should be of type int64)
                    if(ASRUtils::extract_kind_from_ttype_t(
                        ASRUtils::expr_type(m_dims[i].m_length)) > 4){
                        tmp = builder->CreateTrunc(tmp, llvm::IntegerType::get(context, 32));
                    } else if (ASRUtils::extract_kind_from_ttype_t(
                        ASRUtils::expr_type(m_dims[i].m_length)) < 4){
                        tmp = builder->CreateSExt(tmp, llvm::IntegerType::get(context, 32));
                    }

                    array_size = builder->CreateMul(array_size, tmp);
                }
                ptr_loads = ptr_loads_copy;
            }
            llvm::Value *ptr = nullptr;
            // Allocate the variable
            if( !compiler_options.stack_arrays && array_size ) { // malloc for PointerToDataArray
                if(is_array_of_strings){
                    ptr = create_and_setup_string_for_array(v->m_type, array_size, compiler_options.stack_arrays, v->m_name);
                } else {
                    llvm::DataLayout data_layout(module->getDataLayout());
                    uint64_t size = data_layout.getTypeAllocSize(type);
                    array_size = builder->CreateMul(array_size,
                        llvm::ConstantInt::get(context, llvm::APInt(32, size)));
                    llvm::Value* ptr_i8 = LLVMArrUtils::lfortran_malloc(
                        context, *module, *builder, array_size);
                    heap_arrays.push_back(ptr_i8);
                    ptr = builder->CreateBitCast(ptr_i8, type->getPointerTo());
                }
            } else if(ASRUtils::is_string_only(v->m_type)){
                if(v->m_storage == ASR::storage_typeType::Save){
                    std::string str_initial_value_string{};
                    if(v->m_symbolic_value){ // Get initial value if exist.
                        char* str_inital_value{};
                        ASRUtils::extract_value(v->m_symbolic_value, str_inital_value);
                        int str_initial_value_len{};
                        ASRUtils::extract_value(
                            ASRUtils::get_string_type(ASRUtils::expr_type(v->m_symbolic_value))->m_len,
                            str_initial_value_len);
                        str_initial_value_string = std::string(str_inital_value, str_initial_value_len);
                    }
                    ptr = llvm_utils->declare_global_string(ASRUtils::get_string_type(v->m_type),
                            str_initial_value_string, false,
                            v->m_name);
                } else {
                    // Create String + Set it up
                    ptr = llvm_utils->create_string(ASRUtils::get_string_type(v->m_type), v->m_name);
                    setup_string(ptr, v->m_type);
                }
            } else { // Alloca for rest of types (not its internals if exist).
                if (v->m_storage == ASR::storage_typeType::Save) {
                    std::string parent_function_name = std::string(x.m_name);
                    std::string global_name = parent_function_name+ "." + v->m_name;
                    ptr = module->getOrInsertGlobal(global_name, type);
                    llvm::GlobalVariable *gptr = module->getNamedGlobal(global_name);
                    gptr->setLinkage(llvm::GlobalValue::InternalLinkage);
                    llvm::Constant *init_value;
                    if (v->m_value
                            && (ASR::is_a<ASR::Integer_t>(*v->m_type)
                            || ASR::is_a<ASR::Real_t>(*v->m_type)
                            || ASR::is_a<ASR::Logical_t>(*v->m_type))) {
                        this->visit_expr(*v->m_value);
                        init_value = llvm::dyn_cast<llvm::Constant>(tmp);
                    } else {
                        init_value = llvm::Constant::getNullValue(type);
                    }
                    gptr->setInitializer(init_value);
                    #if LLVM_VERSION_MAJOR > 16
                        ptr_type[ptr] = type;
                    #endif
                } else {
#if LLVM_VERSION_MAJOR > 16
                    bool is_llvm_ptr = false;
                    if ( LLVM::is_llvm_pointer(*v->m_type) &&
                            !ASRUtils::is_class_type(ASRUtils::type_get_past_pointer(
                            ASRUtils::type_get_past_allocatable(v->m_type))) &&
                            !ASRUtils::is_descriptorString(v->m_type) ) {
                        is_llvm_ptr = true;
                    }
                    ptr = llvm_utils->CreateAlloca(*builder, type_, array_size,
                        v->m_name, is_llvm_ptr);
#else
                    ptr = llvm_utils->CreateAlloca(*builder, type, array_size, v->m_name);
#endif
                }
            }
            set_pointer_variable_to_null(v, llvm::ConstantPointerNull::get(
                static_cast<llvm::PointerType*>(type)), ptr);
            ASR::expr_t* var_expr = ASRUtils::EXPR(ASR::make_Var_t(al, v->base.base.loc, &v->base));
            // Initialize non-primitve types
            if( ASR::is_a<ASR::StructType_t>(
                *ASRUtils::type_get_past_array(v->m_type))
                && !ASRUtils::is_class_type(
                    ASRUtils::type_get_past_array(v->m_type))) {
                if( ASRUtils::is_array(v->m_type) ) {
                    allocate_array_members_of_struct_arrays(var_expr, ptr, v->m_type);
                } else {
                    allocate_array_members_of_struct(ASR::down_cast<ASR::Struct_t>(
                        ASRUtils::symbol_get_past_external(ASRUtils::get_struct_sym_from_struct_expr(var_expr))), ptr, v->m_type);
                }
            }
            if (compiler_options.emit_debug_info) {
                // Reset the debug location
                builder->SetCurrentDebugLocation(nullptr);
                uint32_t line, column;
                if (compiler_options.emit_debug_line_column) {
                    debug_get_line_column(v->base.base.loc.first, line, column);
                } else {
                    line = v->base.base.loc.first;
                    column = 0;
                }
                std::string type_name;
                uint32_t type_size, type_encoding;
                get_type_debug_info(v->m_type, type_name, type_size,
                    type_encoding);
                llvm::DILocalVariable *debug_var = DBuilder->createParameterVariable(
                    debug_current_scope, v->m_name, ++debug_arg_count, debug_Unit, line,
                    DBuilder->createBasicType(type_name, type_size, type_encoding), true);
                DBuilder->insertDeclare(ptr, debug_var, DBuilder->createExpression(),
                    llvm::DILocation::get(debug_current_scope->getContext(),
                    line, 0, debug_current_scope), builder->GetInsertBlock());
            }

            if( ASR::is_a<ASR::StructType_t>(*v->m_type) && !ASRUtils::is_class_type(v->m_type) ) {
                ASR::Struct_t* struct_type = ASR::down_cast<ASR::Struct_t>(ASRUtils::symbol_get_past_external(v->m_type_declaration));
                int64_t alignment_value = -1;
                if( ASRUtils::extract_value(struct_type->m_alignment, alignment_value) ) {
                    llvm::Align align(alignment_value);
                    reinterpret_cast<llvm::AllocaInst*>(ptr)->setAlignment(align);
                }
            }
            llvm_symtab[h] = ptr;
            if( (ASRUtils::is_array(v->m_type) &&
                ((ASRUtils::extract_physical_type(v->m_type) == ASR::array_physical_typeType::DescriptorArray) ||
                (ASRUtils::extract_physical_type(v->m_type) == ASR::array_physical_typeType::StringArraySinglePointer &&
                ASRUtils::is_dimension_empty(m_dims,n_dims))))
            ) {
                fill_array_details_(ASRUtils::EXPR(ASR::make_Var_t(
                    al, v->base.base.loc, &v->base)), ptr, type_, m_dims, n_dims,
                    is_malloc_array_type, is_array_type, is_list, v->m_type);
            }
            ASR::expr_t* init_expr = v->m_symbolic_value;
            if( v->m_storage != ASR::storage_typeType::Parameter ) {
                for( size_t i = 0; i < v->n_dependencies; i++ ) {
                    std::string variable_name = v->m_dependencies[i];
                    ASR::symbol_t* dep_sym = x.m_symtab->resolve_symbol(variable_name);
                    if (dep_sym) {
                        if (ASR::is_a<ASR::Variable_t>(*dep_sym)) {
                            ASR::Variable_t* dep_v = ASR::down_cast<ASR::Variable_t>(dep_sym);
                            if ( dep_v->m_symbolic_value == nullptr &&
                                !(ASRUtils::is_array(dep_v->m_type) && ASRUtils::extract_physical_type(dep_v->m_type) ==
                                    ASR::array_physical_typeType::FixedSizeArray)) {
                                init_expr = nullptr;
                                break;
                            }
                        }
                    }
                }
            }
            if( init_expr != nullptr && !is_list && !is_dict) {
                target_var = ptr;
                if (v->m_storage == ASR::storage_typeType::Save && 
                    ASRUtils::is_string_only(v->m_type)) {
                    // DO Nothing 
                    // (String + Save) variable is declared as global llvm variable with the intended inital value
                } else if(v->m_storage == ASR::storage_typeType::Save &&
                    ASR::is_a<ASR::Function_t>(
                        *ASR::down_cast<ASR::symbol_t>(v->m_parent_symtab->asr_owner))){
                    variable_inital_value_vec.push_back({v, target_var});
                } else {
                    set_VariableInital_value(v, target_var);
                }
            } else {
                if (is_list) {
                    ASR::List_t* asr_list = ASR::down_cast<ASR::List_t>(v->m_type);
                    std::string type_code = ASRUtils::get_type_code(asr_list->m_type);
                    list_api->list_init(type_code, ptr, module.get());
                } else if (is_dict) {
                    ASR::Dict_t* asr_dict = ASR::down_cast<ASR::Dict_t>(v->m_type);
                    std::string key_type_code = ASRUtils::get_type_code(asr_dict->m_key_type);
                    std::string value_type_code = ASRUtils::get_type_code(asr_dict->m_value_type);

                    llvm_utils->get_type_from_ttype_t_util(nullptr, ASRUtils::TYPE(ASR::make_List_t(al, asr_dict->base.base.loc, asr_dict->m_key_type)), module.get());
                    llvm_utils->get_type_from_ttype_t_util(nullptr, ASRUtils::TYPE(ASR::make_List_t(al, asr_dict->base.base.loc, asr_dict->m_value_type)), module.get());
                    llvm_utils->get_type_from_ttype_t_util(nullptr, v->m_type, module.get());

                    if (ASRUtils::is_character(*asr_dict->m_key_type))
                        dict_api_sc->dict_init(key_type_code, value_type_code, ptr, module.get(), 0);
                    else
                        dict_api_lp->dict_init(key_type_code, value_type_code, ptr, module.get(), 0);
                }
            }
        }
    }

    template<typename T>
    void declare_vars(const T &x, bool create_vtabs=true) {
        uint32_t debug_arg_count = 0;
        std::vector<std::string> var_order = ASRUtils::determine_variable_declaration_order(x.m_symtab);
        if( create_vtabs ) {
            std::set<std::string> variable_type_names;
            std::vector<ASR::symbol_t*> struct_types;
            // Collects all Struct symbols and StructType variables type names in x.m_symtab and its parent symtabs
            collect_variable_types_and_struct_types(variable_type_names, struct_types, x.m_symtab);
            for( size_t i = 0; i < struct_types.size(); i++ ) {
                ASR::symbol_t* struct_type = struct_types[i];
                bool create_vtab = false;
                for( const std::string& variable_type_name: variable_type_names ) {
                    ASR::Variable_t* var = ASR::down_cast<ASR::Variable_t>(
                        ASRUtils::symbol_get_past_external(x.m_symtab->resolve_symbol(variable_type_name)));
                    ASR::symbol_t* class_sym = ASRUtils::symbol_get_past_external(var->m_type_declaration);
                    if (ASRUtils::is_allocatable(var->m_type) && ASRUtils::is_class_type(ASRUtils::type_get_past_allocatable(var->m_type))) {
                        create_vtab = true;
                        break;
                    }
                    bool is_vtab_needed = false;
                    while( !is_vtab_needed && struct_type ) {
                        if( struct_type == class_sym ) {
                            is_vtab_needed = true;
                        } else {
                            struct_type = ASR::down_cast<ASR::Struct_t>(
                                ASRUtils::symbol_get_past_external(struct_type))->m_parent;
                        }
                    }
                    if( is_vtab_needed ) {
                        create_vtab = true;
                        break;
                    }
                }
                if( create_vtab ) {
                    create_vtab_for_struct_type(
                        ASRUtils::symbol_get_past_external(struct_types[i]),
                        x.m_symtab);
                }
            }
        }
        for (auto &item : var_order) {
            ASR::symbol_t* var_sym = x.m_symtab->get_symbol(item);
            if (is_a<ASR::Variable_t>(*var_sym)) {
                process_Variable(var_sym, x, debug_arg_count);
            }
        }
    }

    bool is_function_variable(const ASR::Variable_t &v) {
        if (v.m_type_declaration) {
            return ASR::is_a<ASR::Function_t>(*ASRUtils::symbol_get_past_external(v.m_type_declaration));
        } else {
            return false;
        }
    }

    bool is_function_variable(const ASR::symbol_t *v) {
        if( !ASR::is_a<ASR::Variable_t>(*v) ) {
            return false;
        }
        return is_function_variable(*ASR::down_cast<ASR::Variable_t>(v));
    }

    // F is the function that we are generating and we go over all arguments
    // (F.args()) and handle three cases:
    //     * Variable (`integer :: x`)
    //     * Function (callback) Variable (`procedure(fn) :: x`)
    //     * Function (`fn`)
    void declare_args(const ASR::Function_t &x, llvm::Function &F) {
        size_t i = 0;
        for (llvm::Argument &llvm_arg : F.args()) {
            ASR::symbol_t *s = symbol_get_past_external(
                    ASR::down_cast<ASR::Var_t>(x.m_args[i])->m_v);
            ASR::symbol_t* arg_sym = s;
            if (is_a<ASR::Variable_t>(*s)) {
                ASR::Variable_t *v = ASR::down_cast<ASR::Variable_t>(s);
                if (is_function_variable(*v)) {
                    // * Function (callback) Variable (`procedure(fn) :: x`)
                    s = ASRUtils::symbol_get_past_external(v->m_type_declaration);
                } else {
                    // * Variable (`integer :: x`)
                    ASR::Variable_t *arg = EXPR2VAR(x.m_args[i]);
                    LCOMPILERS_ASSERT(is_arg_dummy(arg->m_intent));

                    llvm::Value* llvm_sym = &llvm_arg;
                    uint32_t h = get_hash((ASR::asr_t*)arg);
                    std::string arg_s = arg->m_name;
                    llvm_arg.setName(arg_s);
                    llvm_symtab[h] = llvm_sym;
#if LLVM_VERSION_MAJOR > 16
                    llvm::Type *arg_type = llvm_utils->get_type_from_ttype_t_util(x.m_args[i],
                        ASRUtils::type_get_past_allocatable(
                        ASRUtils::type_get_past_pointer(arg->m_type)), module.get());
                    if ( !ASRUtils::is_array(arg->m_type) &&
                        LLVM::is_llvm_pointer(*arg->m_type) &&
                        !ASRUtils::is_class_type(ASRUtils::type_get_past_allocatable(
                            ASRUtils::type_get_past_pointer(arg->m_type))) ) {
                        arg_type = arg_type->getPointerTo();
                    }
                    ptr_type[llvm_sym] = arg_type;
#endif
                }
            }
            if (is_a<ASR::Function_t>(*s)) {
                // * Function (`fn`)
                // Deal with case where procedure passed in as argument
                ASR::Function_t *arg = ASR::down_cast<ASR::Function_t>(s);
                uint32_t h = get_hash((ASR::asr_t*)arg_sym);
                std::string arg_s = ASRUtils::symbol_name(arg_sym);
                llvm_arg.setName(arg_s);
                llvm_symtab_fn_arg[h] = &llvm_arg;
                if( is_function_variable(arg_sym) ) {
                    llvm_symtab[h] = &llvm_arg;
                }
                if (llvm_symtab_fn.find(h) == llvm_symtab_fn.end()) {
                    llvm::FunctionType* fntype = llvm_utils->get_function_type(*arg, module.get());
                    llvm::Function* fn = llvm::Function::Create(fntype, llvm::Function::ExternalLinkage, arg->m_name, module.get());
                    llvm_symtab_fn[h] = fn;
                }
            }
            i++;
        }
    }

    template <typename T>
    void declare_local_vars(const T &x) {
        declare_vars(x);
    }

    void visit_Function(const ASR::Function_t &x) {
        loop_head.clear();
        loop_head_names.clear();
        loop_or_block_end.clear();
        loop_or_block_end_names.clear();
        heap_arrays.clear();
        strings_to_be_deallocated.reserve(al, 1);
        SymbolTable* current_scope_copy = current_scope;
        current_scope = x.m_symtab;
        bool is_dict_present_copy_lp = dict_api_lp->is_dict_present();
        bool is_dict_present_copy_sc = dict_api_sc->is_dict_present();
        dict_api_lp->set_is_dict_present(false);
        dict_api_sc->set_is_dict_present(false);
        bool is_set_present_copy_lp = set_api_lp->is_set_present();
        bool is_set_present_copy_sc = set_api_sc->is_set_present();
        set_api_lp->set_is_set_present(false);
        set_api_sc->set_is_set_present(false);
        llvm_goto_targets.clear();
        instantiate_function(x);
        if (ASRUtils::get_FunctionType(x)->m_deftype == ASR::deftypeType::Interface) {
            // Interface does not have an implementation and it is already
            // declared, so there is nothing to do here
            return;
        }
        visit_procedures(x);
        generate_function(x);
        parent_function = nullptr;
        dict_api_lp->set_is_dict_present(is_dict_present_copy_lp);
        dict_api_sc->set_is_dict_present(is_dict_present_copy_sc);
        set_api_lp->set_is_set_present(is_set_present_copy_lp);
        set_api_sc->set_is_set_present(is_set_present_copy_sc);

        // Finalize the debug info.
        if (compiler_options.emit_debug_info) DBuilder->finalize();
        current_scope = current_scope_copy;
        loop_head.clear();
        loop_head_names.clear();
        loop_or_block_end.clear();
        loop_or_block_end_names.clear();
        heap_arrays.clear();
        strings_to_be_deallocated.reserve(al, 1);
    }

    void instantiate_function(const ASR::Function_t &x){
        uint32_t h = get_hash((ASR::asr_t*)&x);
        llvm::Function *F = nullptr;
        llvm::DISubprogram *SP = nullptr;
        std::string sym_name = x.m_name;
        if (sym_name == "main") {
            sym_name = "_xx_lcompilers_changed_main_xx";
        }
        if (llvm_symtab_fn.find(h) != llvm_symtab_fn.end()) {
            /*
            throw CodeGenError("Function code already generated for '"
                + std::string(x.m_name) + "'");
            */
            F = llvm_symtab_fn[h];
        } else {
            llvm::FunctionType* function_type = llvm_utils->get_function_type(x, module.get());
            if( ASRUtils::get_FunctionType(x)->m_deftype == ASR::deftypeType::Interface && !ASRUtils::get_FunctionType(x)->m_module ) {
                ASR::FunctionType_t* asr_function_type = ASRUtils::get_FunctionType(x);
                for( size_t i = 0; i < asr_function_type->n_arg_types; i++ ) {
                    if( ASRUtils::is_class_type(asr_function_type->m_arg_types[i]) ) {
                        return ;
                    }
                }
            }
            std::string fn_name;
            if (ASRUtils::get_FunctionType(x)->m_abi == ASR::abiType::BindC) {
                if (ASRUtils::get_FunctionType(x)->m_bindc_name) {
                    fn_name = ASRUtils::get_FunctionType(x)->m_bindc_name;
                } else {
                    fn_name = sym_name;
                }
            } else if (ASRUtils::get_FunctionType(x)->m_deftype == ASR::deftypeType::Interface &&
                ASRUtils::get_FunctionType(x)->m_abi != ASR::abiType::Intrinsic && !ASRUtils::get_FunctionType(x)->m_module) {
                fn_name = sym_name;
            } else {
                fn_name = mangle_prefix + sym_name;
            }
            if ( parent_function != nullptr &&
                 ASRUtils::get_FunctionType(x)->m_deftype != ASR::deftypeType::Interface &&
                 ASRUtils::get_FunctionType(x)->m_abi != ASR::abiType::Intrinsic &&
                 ASRUtils::get_FunctionType(x)->m_abi != ASR::abiType::BindC
                ) {
                std::string parent_function_name = std::string(parent_function->m_name);
                fn_name = parent_function_name+ "." + fn_name;
            }
            if (llvm_symtab_fn_names.find(fn_name) == llvm_symtab_fn_names.end()) {
                llvm_symtab_fn_names[fn_name] = h;
                F = llvm::Function::Create(function_type,
                    llvm::Function::ExternalLinkage, fn_name, module.get());

                // Add Debugging information to the LLVM function F
                if (compiler_options.emit_debug_info) {
                    debug_emit_function(x, SP);
                    F->setSubprogram(SP);
                }
            } else {
                uint32_t old_h = llvm_symtab_fn_names[fn_name];
                F = llvm_symtab_fn[old_h];
                if (compiler_options.emit_debug_info) {
                    SP = (llvm::DISubprogram*) llvm_symtab_fn_discope[old_h];
                }
            }
            llvm_symtab_fn[h] = F;
            if (compiler_options.emit_debug_info) llvm_symtab_fn_discope[h] = SP;

            // Instantiate (pre-declare) all nested interfaces
            for (auto &item : x.m_symtab->get_scope()) {
                if (is_a<ASR::Function_t>(*item.second)) {
                    const ASR::Function_t* parent_function_copy = parent_function;
                    parent_function = &x;
                    ASR::Function_t *v = down_cast<ASR::Function_t>(
                            item.second);
                    // check if item.second is present in x.m_args
                    bool interface_as_arg = false;
                    for (size_t i=0; i<x.n_args; i++) {
                        if (is_a<ASR::Var_t>(*x.m_args[i])) {
                            ASR::Var_t *arg = down_cast<ASR::Var_t>(x.m_args[i]);
                            if ( arg->m_v == item.second ) {
                                interface_as_arg = true;
                                llvm::FunctionType* fntype = llvm_utils->get_function_type(*v, module.get());
                                llvm::Function* fn = llvm::Function::Create(fntype, llvm::Function::ExternalLinkage, v->m_name, module.get());
                                uint32_t hash = get_hash((ASR::asr_t*)v);
                                llvm_symtab_fn[hash] = fn;
                            }
                        }
                    }
                    if (!interface_as_arg) {
                        instantiate_function(*v);
                    }
                    parent_function = parent_function_copy;
                } else if ( ASR::is_a<ASR::Variable_t>(*ASRUtils::symbol_get_past_external(item.second)) && is_function_variable(ASRUtils::symbol_get_past_external(item.second)) ) {
                    ASR::Variable_t *v = down_cast<ASR::Variable_t>(ASRUtils::symbol_get_past_external(item.second));
                    bool interface_as_arg = false;
                    for (size_t i=0; i<x.n_args; i++) {
                        if (is_a<ASR::Var_t>(*x.m_args[i])) {
                            ASR::Var_t *arg = down_cast<ASR::Var_t>(x.m_args[i]);
                            if ( arg->m_v == item.second ) {
                                interface_as_arg = true;
                            }
                        }
                    }
                    if ( interface_as_arg ) {
                        continue;
                    }
                    ASR::Function_t *var = ASR::down_cast<ASR::Function_t>(
                            ASRUtils::symbol_get_past_external(v->m_type_declaration));
                    uint32_t h = get_hash((ASR::asr_t*)v);
                    if (llvm_symtab_fn.find(h) != llvm_symtab_fn.end()) {
                        continue;
                    }
                    llvm::FunctionType* function_type = llvm_utils->get_function_type(*var, module.get());
                    std::string fn_name;
                    std::string sym_name = v->m_name;
                    if (ASRUtils::get_FunctionType(*var)->m_abi == ASR::abiType::BindC) {
                        if (ASRUtils::get_FunctionType(*var)->m_bindc_name) {
                            fn_name = ASRUtils::get_FunctionType(*var)->m_bindc_name;
                        } else {
                            fn_name = sym_name;
                        }
                    } else if (ASRUtils::get_FunctionType(*var)->m_deftype == ASR::deftypeType::Interface &&
                        ASRUtils::get_FunctionType(*var)->m_abi != ASR::abiType::Intrinsic) {
                        fn_name = sym_name;
                    } else {
                        fn_name = mangle_prefix + sym_name;
                    }
                    if (llvm_symtab_fn_names.find(fn_name) == llvm_symtab_fn_names.end()) {
                        llvm_symtab_fn_names[fn_name] = h;
                        llvm::Function* F = llvm::Function::Create(function_type,
                            llvm::Function::ExternalLinkage, fn_name, module.get());
                        llvm_symtab_fn[h] = F;
                    } else {
                        uint32_t old_h = llvm_symtab_fn_names[fn_name];
                        llvm_symtab_fn[h] = llvm_symtab_fn[old_h];
                    }
                }
            }
        }
    }

    inline void define_function_entry(const ASR::Function_t& x) {
        uint32_t h = get_hash((ASR::asr_t*)&x);
        parent_function = &x;
        llvm::Function* F = llvm_symtab_fn[h];
        if (compiler_options.emit_debug_info) debug_current_scope = llvm_symtab_fn_discope[h];
        proc_return = llvm::BasicBlock::Create(context, "return");
        llvm::BasicBlock *BB = llvm::BasicBlock::Create(context,
                ".entry", F);
        builder->SetInsertPoint(BB);
        if (compiler_options.emit_debug_info) debug_emit_loc(x);
        declare_args(x, *F);
        for( auto& sym: x.m_symtab->get_scope() ) {
            if( !ASR::is_a<ASR::Variable_t>(*sym.second) ) {
                continue ;
            }
            ASR::intentType symbol_intent = ASRUtils::symbol_intent(
                ASRUtils::symbol_get_past_external(sym.second));
            if( !(symbol_intent == ASRUtils::intent_in ||
                 symbol_intent == ASRUtils::intent_inout ||
                 symbol_intent == ASRUtils::intent_out) ) {
                continue;
            }
            ASR::ttype_t* symbol_type = ASRUtils::symbol_type(sym.second);
            // Reinitialize StructType members if intent is intent_out
            if (ASR::is_a<ASR::StructType_t>(*symbol_type) && symbol_intent == ASRUtils::intent_out) {
                uint32_t h = get_hash((ASR::asr_t*)sym.second);
                LCOMPILERS_ASSERT(llvm_symtab.find(h) != llvm_symtab.end());
                llvm::Value* st_desc = llvm_symtab[h];

                ASR::Struct_t* struct_sym = ASR::down_cast<ASR::Struct_t>(ASRUtils::symbol_get_past_external(
                        ASR::down_cast<ASR::Variable_t>(sym.second)->m_type_declaration));

                if (ASRUtils::is_class_type(symbol_type)) {
                    // TODO: Make this work for extended types too
                    llvm::Type* src_class_type = llvm_utils->get_type_from_ttype_t_util(
                        ASRUtils::EXPR(ASR::make_Var_t(al, x.base.base.loc, sym.second)),
                        symbol_type, module.get());
                    st_desc = llvm_utils->create_gep2(src_class_type, st_desc, 1);

                    ASR::Struct_t* src_struct_sym = ASR::down_cast<ASR::Struct_t>(
                            ASRUtils::symbol_get_past_external(ASRUtils::get_struct_sym_from_struct_expr(ASRUtils::EXPR(ASR::make_Var_t(al, x.base.base.loc, sym.second)))));
                    llvm::Type* src_struct_type = get_llvm_struct_data_type(src_struct_sym, true);
                    st_desc = llvm_utils->CreateLoad2(src_struct_type, st_desc);

                    allocate_array_members_of_struct(struct_sym, st_desc, ASR::down_cast<ASR::Variable_t>(sym.second)->m_type, true);
                } else {
                    allocate_array_members_of_struct(struct_sym, st_desc, ASR::down_cast<ASR::Variable_t>(sym.second)->m_type, true);
                }
            }
            if( !(ASRUtils::is_pointer(symbol_type) || ASRUtils::is_allocatable(symbol_type)) &&
                ASRUtils::is_array(symbol_type) &&
                ASRUtils::extract_physical_type(symbol_type)
                    == ASR::array_physical_typeType::DescriptorArray ) {
                llvm::Type* desc_array_type = llvm_utils->get_type_from_ttype_t_util(ASRUtils::get_expr_from_sym(al, sym.second),
                        ASRUtils::type_get_past_allocatable_pointer(symbol_type),
                        module.get());
                llvm::Value* array_desc = llvm_utils->CreateAlloca(
                    desc_array_type, nullptr, "array_descriptor_local");
                uint32_t h = get_hash((ASR::asr_t*)sym.second);
                LCOMPILERS_ASSERT(llvm_symtab.find(h) != llvm_symtab.end());
                llvm::Value* arg_array_desc = llvm_symtab[h];
                llvm::Type *data_type = llvm_utils->get_type_from_ttype_t_util(ASRUtils::get_expr_from_sym(al, sym.second),
                    ASRUtils::extract_type(symbol_type), module.get());
                builder->CreateStore(llvm_utils->create_ptr_gep2(data_type,
                llvm_utils->CreateLoad2(data_type->getPointerTo(), arr_descr->get_pointer_to_data(arg_array_desc)),
                    arr_descr->get_offset(arg_array_desc)), arr_descr->get_pointer_to_data(array_desc));
                ASR::dimension_t* m_dims = nullptr;
                int n_dims = ASRUtils::extract_dimensions_from_ttype(symbol_type, m_dims);
                Vec<llvm::Value*> lbs, lengths;
                lbs.reserve(al, n_dims);
                lengths.reserve(al, n_dims);
                int64_t ptr_loads_copy = ptr_loads;
                ptr_loads = 2;
                for( int i = 0; i < n_dims; i++ ) {
                    if( m_dims[i].m_start ) {
                        visit_expr_wrapper(m_dims[i].m_start);
                        lbs.push_back(al, tmp);
                    } else {
                        lbs.push_back(al, nullptr);
                    }
                    if( m_dims[i].m_length ) {
                        visit_expr_wrapper(m_dims[i].m_length);
                        lengths.push_back(al, tmp);
                    } else {
                        lengths.push_back(al, nullptr);
                    }
                }
                arr_descr->reset_array_details(
                    array_desc, arg_array_desc, lbs.p, lengths.p, n_dims);
                ptr_loads = ptr_loads_copy;
                llvm_symtab[h] = array_desc;
            }
        }
        declare_local_vars(x);
    }


    inline void define_function_exit(const ASR::Function_t& x) {
        if (x.m_return_var) {
            start_new_block(proc_return);
            ASR::Variable_t *asr_retval = EXPR2VAR(x.m_return_var);
            uint32_t h = get_hash((ASR::asr_t*)asr_retval);
            llvm::Value *ret_val = llvm_symtab[h];
            llvm::Value *ret_val2 = ret_val;
            if (!ASRUtils::is_class_type(ASRUtils::extract_type(asr_retval->m_type)) &&
                !(ASR::is_a<ASR::Pointer_t>(*asr_retval->m_type) && ASR::is_a<ASR::String_t>(*ASRUtils::extract_type(asr_retval->m_type))) ) {
                llvm::Type* asr_retval_llvm_type = llvm_utils->get_type_from_ttype_t_util(ASRUtils::EXPR(ASR::make_Var_t(
                    al, asr_retval->base.base.loc, &asr_retval->base)), asr_retval->m_type, module.get());
                ret_val2 = llvm_utils->CreateLoad2(asr_retval_llvm_type, ret_val);
            }
            // Handle Complex type return value for BindC:
            if (ASRUtils::get_FunctionType(x)->m_abi == ASR::abiType::BindC) {
                ASR::ttype_t* arg_type = asr_retval->m_type;
                llvm::Value *tmp = ret_val;
                if (is_a<ASR::Complex_t>(*arg_type)) {
                    int c_kind = ASRUtils::extract_kind_from_ttype_t(arg_type);
                    if (c_kind == 4) {
                        if (compiler_options.platform == Platform::Windows) {
                            // tmp is {float, float}*
                            // type_fx2p is i64*
                            llvm::Type* type_fx2p = llvm::Type::getInt64Ty(context)->getPointerTo();
                            // Convert {float,float}* to i64* using bitcast
                            tmp = builder->CreateBitCast(tmp, type_fx2p);
                            // Then convert i64* -> i64
                            tmp = llvm_utils->CreateLoad(tmp);
                        } else if (compiler_options.platform == Platform::macOS_ARM) {
                            // Pass by value
                            tmp = llvm_utils->CreateLoad(tmp);
                        } else {
                            // tmp is {float, float}*
                            // type_fx2 is <2 x float>
                            llvm::Type* type_fx2 = FIXED_VECTOR_TYPE::get(llvm::Type::getFloatTy(context), 2);
                            // Convert {float,float}* to <2 x float>* using bitcast
                            tmp = builder->CreateBitCast(tmp, type_fx2->getPointerTo());
                            // Then convert <2 x float>* -> <2 x float>
                            tmp = llvm_utils->CreateLoad2(type_fx2, tmp);
                        }
                    } else {
                        LCOMPILERS_ASSERT(c_kind == 8)
                        if (compiler_options.platform == Platform::Windows) {
                            // 128 bit aggregate type is passed by reference
                        } else {
                            // Pass by value
                            tmp = llvm_utils->CreateLoad(tmp);
                        }
                    }
                ret_val2 = tmp;
                }
            }
            for( auto& value: heap_arrays ) {
                LLVM::lfortran_free(context, *module, *builder, value);
            }
            builder->CreateRet(ret_val2);
        } else {
            start_new_block(proc_return);
            for( auto& value: heap_arrays ) {
                LLVM::lfortran_free(context, *module, *builder, value);
            }
            builder->CreateRetVoid();
        }
    }

    void generate_function(const ASR::Function_t &x) {
        bool interactive = (ASRUtils::get_FunctionType(x)->m_abi == ASR::abiType::ExternalUndefined);
        if (ASRUtils::get_FunctionType(x)->m_deftype == ASR::deftypeType::Implementation ) {

            if (interactive) return;

            if (!prototype_only) {
                define_function_entry(x);

                for (size_t i=0; i<x.n_body; i++) {
                    this->visit_stmt(*x.m_body[i]);
                }

                define_function_exit(x);
            }
        } else if( ASRUtils::get_FunctionType(x)->m_abi == ASR::abiType::Intrinsic &&
                   ASRUtils::get_FunctionType(x)->m_deftype == ASR::deftypeType::Interface ) {
            std::string m_name = x.m_name;
            if( m_name == "lbound" || m_name == "ubound" ) {
                define_function_entry(x);

                // Defines the size intrinsic's body at LLVM level.
                ASR::Variable_t *arg = EXPR2VAR(x.m_args[0]);
                uint32_t h = get_hash((ASR::asr_t*)arg);
                llvm::Value* llvm_arg1 = llvm_symtab[h];

                arg = EXPR2VAR(x.m_args[1]);
                h = get_hash((ASR::asr_t*)arg);
                llvm::Value* llvm_arg2 = llvm_symtab[h];

                ASR::Variable_t *ret = EXPR2VAR(x.m_return_var);
                h = get_hash((ASR::asr_t*)ret);
                llvm::Value* llvm_ret_ptr = llvm_symtab[h];

                llvm::Value* dim_des_val = llvm_utils->CreateLoad(llvm_arg1);
                llvm::Value* dim_val = llvm_utils->CreateLoad(llvm_arg2);
                llvm::Value* const_1 = llvm::ConstantInt::get(context, llvm::APInt(32, 1));
                dim_val = builder->CreateSub(dim_val, const_1);
                llvm::Value* dim_struct = arr_descr->get_pointer_to_dimension_descriptor(dim_des_val, dim_val);
                llvm::Value* res = nullptr;
                if( m_name == "lbound" ) {
                    res = arr_descr->get_lower_bound(dim_struct);
                } else if( m_name == "ubound" ) {
                    res = arr_descr->get_upper_bound(dim_struct);
                }
                builder->CreateStore(res, llvm_ret_ptr);

                define_function_exit(x);
            }
        }
    }


    template<typename T>
    void visit_procedures(const T &x) {
        for (auto &item : x.m_symtab->get_scope()) {
            if (is_a<ASR::Function_t>(*item.second)) {
                ASR::Function_t *s = ASR::down_cast<ASR::Function_t>(item.second);
                visit_Function(*s);
            }
        }
    }

    bool is_nested_pointer(llvm::Value* val) {
        // TODO: Remove this in future
        // Related issue, https://github.com/lcompilers/lpython/pull/707#issuecomment-1169773106.
        return val->getType()->isPointerTy() &&
               val->getType()->getContainedType(0)->isPointerTy();
    }

    void visit_CLoc(const ASR::CLoc_t& x) {
        int64_t ptr_loads_copy = ptr_loads;
        ptr_loads = 0;
        this->visit_expr(*x.m_arg);
        ptr_loads = ptr_loads_copy;
        if( is_nested_pointer(tmp) ) {
            tmp = llvm_utils->CreateLoad(tmp);
        }
        ASR::ttype_t* arg_type = ASRUtils::get_contained_type(ASRUtils::expr_type(x.m_arg));
        if( ASRUtils::is_array(arg_type) ) {
            tmp = llvm_utils->CreateLoad(arr_descr->get_pointer_to_data(tmp));
        }
        tmp = builder->CreateBitCast(tmp,
                    llvm::Type::getVoidTy(context)->getPointerTo());
    }


    llvm::Value* GetPointerCPtrUtil(llvm::Value* llvm_tmp, ASR::expr_t* asr_expr) {
        // If the input is a simple variable and not a pointer
        // then this check will fail and load will not happen
        // (which is what we want for simple variables).
        // For pointers, the actual LLVM variable will be a
        // double pointer, so we need to load one time and then
        // use it later on.
        ASR::ttype_t* asr_type = ASRUtils::expr_type(asr_expr);
        if(ASR::is_a<ASR::Pointer_t>(*asr_type) &&
            (LLVM::is_llvm_pointer(*ASRUtils::type_get_past_pointer(asr_type))
             || ASR::is_a<ASR::Array_t>(*ASRUtils::type_get_past_pointer(asr_type))
             || llvm::isa<llvm::AllocaInst>(llvm_tmp))) {
            llvm_tmp = llvm_utils->CreateLoad(llvm_tmp);
        }
        asr_type = ASRUtils::get_contained_type(asr_type);

        if( ASRUtils::is_array(asr_type) &&
            !ASR::is_a<ASR::CPtr_t>(*asr_type) ) {
            ASR::array_physical_typeType physical_type = ASRUtils::extract_physical_type(asr_type);
            llvm::Type *el_type = llvm_utils->get_type_from_ttype_t_util(asr_expr, ASRUtils::extract_type(asr_type), module.get());
            switch( physical_type ) {
                case ASR::array_physical_typeType::DescriptorArray: {
#if LLVM_VERSION_MAJOR > 16
                    ptr_type[llvm_tmp] = llvm_utils->get_type_from_ttype_t_util(asr_expr, asr_type, module.get());
#endif
                    llvm_tmp = llvm_utils->CreateLoad2(el_type->getPointerTo(), arr_descr->get_pointer_to_data(llvm_tmp));
                    break;
                }
                case ASR::array_physical_typeType::FixedSizeArray: {
                    llvm::Type* type = llvm_utils->get_type_from_ttype_t_util(asr_expr, asr_type, module.get());
                    llvm_tmp = llvm_utils->create_gep2(type, llvm_tmp, 0);
                    break;
                }
                case ASR::array_physical_typeType::PointerToDataArray: {
                    break;
                }
                default: {
                    LCOMPILERS_ASSERT(false);
                }
            }
        }

        // // TODO: refactor this into a function, it is being used a few times
        // llvm::Type *target_type = llvm_tmp->getType();
        // // Create alloca to get a pointer, but do it
        // // at the beginning of the function to avoid
        // // using alloca inside a loop, which would
        // // run out of stack
        // llvm::BasicBlock &entry_block = builder->GetInsertBlock()->getParent()->getEntryBlock();
        // llvm::IRBuilder<> builder0(context);
        // builder0.SetInsertPoint(&entry_block, entry_block.getFirstInsertionPt());
        // llvm::AllocaInst *target = builder0.CreateAlloca(
        //     target_type, nullptr, "call_arg_value_ptr");
        // builder->CreateStore(llvm_tmp, target);
        // llvm_tmp = target;
        return llvm_tmp;
    }

    void visit_GetPointer(const ASR::GetPointer_t& x) {
        int64_t ptr_loads_copy = ptr_loads;
        ptr_loads = 0;
        this->visit_expr(*x.m_arg);
        ptr_loads = ptr_loads_copy;
        tmp = GetPointerCPtrUtil(tmp, x.m_arg);
    }

    void visit_PointerToCPtr(const ASR::PointerToCPtr_t& x) {
        int64_t ptr_loads_copy = ptr_loads;
        ptr_loads = 0;
        this->visit_expr(*x.m_arg);
        ptr_loads = ptr_loads_copy;
        if( !ASR::is_a<ASR::GetPointer_t>(*x.m_arg) ) {
            tmp = GetPointerCPtrUtil(tmp, x.m_arg);
        }
        tmp = builder->CreateBitCast(tmp,
                    llvm::Type::getVoidTy(context)->getPointerTo());
    }


    void visit_CPtrToPointer(const ASR::CPtrToPointer_t& x) {
        ASR::expr_t *cptr = x.m_cptr, *fptr = x.m_ptr, *shape = x.m_shape;
        int reduce_loads = 0;
        if( ASR::is_a<ASR::Var_t>(*cptr) ) {
            ASR::Variable_t* cptr_var = ASRUtils::EXPR2VAR(cptr);
            reduce_loads = cptr_var->m_intent == ASRUtils::intent_in;
        }
        if( ASRUtils::is_array(ASRUtils::expr_type(fptr)) ) {
            int64_t ptr_loads_copy = ptr_loads;
            ptr_loads = 1 - reduce_loads;
            this->visit_expr(*cptr);
            llvm::Value* llvm_cptr = tmp;
            if (ASR::is_a<ASR::StructInstanceMember_t>(*cptr)) {
                // `type(c_ptr)` requires an extra load here
                // TODO: be more explicit about ptr_loads: https://github.com/lfortran/lfortran/issues/4245
                llvm_cptr = llvm_utils->CreateLoad(llvm_cptr);
            }
            ptr_loads = 0;
            this->visit_expr(*fptr);
            llvm::Value* llvm_fptr = tmp;
            ptr_loads = ptr_loads_copy;
            llvm::Value* llvm_shape = nullptr;
            ASR::ttype_t* asr_shape_type = nullptr;
            if( shape ) {
                asr_shape_type = ASRUtils::get_contained_type(ASRUtils::expr_type(shape));
                this->visit_expr(*shape);
                llvm_shape = tmp;
            }
            ASR::ttype_t* fptr_type = ASRUtils::expr_type(fptr);
            llvm::Type* llvm_fptr_type = llvm_utils->get_type_from_ttype_t_util(fptr,
                ASRUtils::get_contained_type(fptr_type), module.get());
            llvm::Value* fptr_array = llvm_utils->CreateAlloca(*builder, llvm_fptr_type);
            builder->CreateStore(llvm::ConstantInt::get(context, llvm::APInt(32, 0)),
                arr_descr->get_offset(fptr_array, false));
            ASR::dimension_t* fptr_dims;
            int fptr_rank = ASRUtils::extract_dimensions_from_ttype(
                                ASRUtils::expr_type(fptr),
                                fptr_dims);
            llvm::Value* llvm_rank = llvm::ConstantInt::get(context, llvm::APInt(32, fptr_rank));
            llvm::Value* dim_des = llvm_utils->CreateAlloca(*builder, arr_descr->get_dimension_descriptor_type(), llvm_rank);
            builder->CreateStore(dim_des, arr_descr->get_pointer_to_dimension_descriptor_array(fptr_array, false));
            arr_descr->set_rank(fptr_array, llvm_rank);
            builder->CreateStore(fptr_array, llvm_fptr);
            llvm_fptr = fptr_array;
            ASR::ttype_t* fptr_data_type = ASRUtils::duplicate_type_without_dims(al, ASRUtils::get_contained_type(fptr_type), fptr_type->base.loc);
            llvm::Type* llvm_fptr_data_type = llvm_utils->get_type_from_ttype_t_util(fptr, fptr_data_type, module.get());
            llvm::Value* fptr_data = arr_descr->get_pointer_to_data(llvm_fptr);
            llvm::Value* fptr_des = arr_descr->get_pointer_to_dimension_descriptor_array(llvm_fptr);
            llvm::Value* shape_data = llvm_shape;
            if( llvm_shape && (ASRUtils::extract_physical_type(asr_shape_type) ==
                ASR::array_physical_typeType::DescriptorArray) ) {
                shape_data = llvm_utils->CreateLoad(arr_descr->get_pointer_to_data(llvm_shape));
            }
            llvm_cptr = builder->CreateBitCast(llvm_cptr, llvm_fptr_data_type->getPointerTo());
            builder->CreateStore(llvm_cptr, fptr_data);
            llvm::Value* prod = llvm::ConstantInt::get(context, llvm::APInt(32, 1));
            ASR::ArrayConstant_t* lower_bounds = nullptr;
            if( x.m_lower_bounds ) {
                LCOMPILERS_ASSERT(ASR::is_a<ASR::ArrayConstant_t>(*x.m_lower_bounds));
                lower_bounds = ASR::down_cast<ASR::ArrayConstant_t>(x.m_lower_bounds);
                LCOMPILERS_ASSERT(fptr_rank == ASRUtils::get_fixed_size_of_array(lower_bounds->m_type));
            }
            for( int i = 0; i < fptr_rank; i++ ) {
                llvm::Value* curr_dim = llvm::ConstantInt::get(context, llvm::APInt(32, i));
                llvm::Value* desi = arr_descr->get_pointer_to_dimension_descriptor(fptr_des, curr_dim);
                llvm::Value* desi_stride = arr_descr->get_stride(desi, false);
                llvm::Value* desi_lb = arr_descr->get_lower_bound(desi, false);
                llvm::Value* desi_size = arr_descr->get_dimension_size(fptr_des, curr_dim, false);
                builder->CreateStore(prod, desi_stride);
                llvm::Value* i32_one = llvm::ConstantInt::get(context, llvm::APInt(32, 1));
                llvm::Value* new_lb = i32_one;
                if( lower_bounds ) {
                    int ptr_loads_copy = ptr_loads;
                    ptr_loads = 2;
                    this->visit_expr_wrapper(ASRUtils::fetch_ArrayConstant_value(al, lower_bounds, i), true);
                    ptr_loads = ptr_loads_copy;
                    new_lb = tmp;
                }
                llvm::Value* new_ub = nullptr;
                if( ASRUtils::extract_physical_type(asr_shape_type) == ASR::array_physical_typeType::DescriptorArray ||
                    ASRUtils::extract_physical_type(asr_shape_type) == ASR::array_physical_typeType::PointerToDataArray ) {
                    new_ub = shape_data ? llvm_utils->CreateLoad2(
                        llvm::Type::getInt32Ty(context), llvm_utils->create_ptr_gep(shape_data, i)) : i32_one;
                } else if( ASRUtils::extract_physical_type(asr_shape_type) == ASR::array_physical_typeType::FixedSizeArray ) {
                    llvm::Type* shape_llvm_type = llvm_utils->get_type_from_ttype_t_util(shape, asr_shape_type, module.get());
                    new_ub = shape_data ? llvm_utils->CreateLoad2(
                        llvm::Type::getInt32Ty(context), llvm_utils->create_gep2(shape_llvm_type, shape_data, i)) : i32_one;
                }
                builder->CreateStore(new_lb, desi_lb);
                llvm::Value* new_size = builder->CreateAdd(builder->CreateSub(new_ub, new_lb), i32_one);
                builder->CreateStore(new_size, desi_size);
                prod = builder->CreateMul(prod, new_size);
            }
        } else {
            int64_t ptr_loads_copy = ptr_loads;
            ptr_loads = 1 - reduce_loads;
            this->visit_expr(*cptr);
            llvm::Value* llvm_cptr = tmp;
            if (ASR::is_a<ASR::StructInstanceMember_t>(*cptr)) {
                // Load the actual pointer value for type(c_ptr) members
                llvm::Type* cptr_llvm_type = llvm::Type::getInt8Ty(context)->getPointerTo(); // void*
                llvm_cptr = llvm_utils->CreateLoad2(cptr_llvm_type, llvm_cptr);
            } else if (ASR::is_a<ASR::ArrayItem_t>(*cptr)) {
                llvm::Type* cptr_llvm_type = llvm_utils->get_type_from_ttype_t_util(cptr, ASRUtils::expr_type(cptr), module.get());
                llvm_cptr = llvm_utils->CreateLoad2(cptr_llvm_type, llvm_cptr);
            }
            ptr_loads = 0;
            this->visit_expr(*fptr);
            llvm::Value* llvm_fptr = tmp;
            ASR::ttype_t* fptr_type = ASRUtils::expr_type(fptr);
            ASR::ttype_t* contained_type = ASRUtils::get_contained_type(fptr_type);

            if (ASR::is_a<ASR::String_t>(*contained_type)) {
                // Character descriptor: struct { i32 len, i8* ptr }
                llvm::Type* i8_ty = llvm::IntegerType::get(context, 8);
                llvm::Type* i8_ptr_ty = llvm::PointerType::get(i8_ty, 0);
                llvm::Type* char_desc_type = llvm::StructType::get(
                    llvm::Type::getInt32Ty(context), i8_ptr_ty);
                llvm::Value* data_ptr = builder->CreateBitCast(llvm_cptr, i8_ptr_ty);
                llvm::Value* fptr_cast = builder->CreateBitCast(llvm_fptr, char_desc_type->getPointerTo());
                llvm::Value* gep_len = llvm_utils->create_gep2(char_desc_type, fptr_cast, 0);  // .len
                llvm::Value* gep_data = llvm_utils->create_gep2(char_desc_type, fptr_cast, 1); // .ptr

                llvm::Value* len_val = llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), 1); // dummy length
                builder->CreateStore(len_val, gep_len);
                builder->CreateStore(data_ptr, gep_data);
                return;
            }
            ptr_loads = ptr_loads_copy;
            llvm::Type* llvm_fptr_type = llvm_utils->get_type_from_ttype_t_util(fptr,
                ASRUtils::get_contained_type(ASRUtils::expr_type(fptr)), module.get());
            llvm_cptr = builder->CreateBitCast(llvm_cptr, llvm_fptr_type->getPointerTo());
            builder->CreateStore(llvm_cptr, llvm_fptr);
        }
    }

    void visit_PointerAssociated(const ASR::PointerAssociated_t& x) {
        if (x.m_value) {
            this->visit_expr_wrapper(x.m_value, true);
            return;
        }
        llvm::AllocaInst *res = llvm_utils->CreateAlloca(
            llvm::Type::getInt1Ty(context), nullptr, "is_associated");
        ASR::ttype_t* p_type = ASRUtils::expr_type(x.m_ptr);
        llvm::Value *ptr, *nptr;
        int64_t ptr_loads_copy = ptr_loads;
        ptr_loads = 0;
        visit_expr_wrapper(x.m_ptr, false);
        ptr = tmp;
#if LLVM_VERSION_MAJOR > 16
        llvm::Type *ptr_arr_type = llvm_utils->get_type_from_ttype_t_util(x.m_ptr,
            ASRUtils::type_get_past_allocatable(
            ASRUtils::type_get_past_pointer(p_type)), module.get());
#endif
        if (ASRUtils::is_class_type(ASRUtils::extract_type(p_type))) {
            // If the pointer is class, get its type pointer
            ptr = llvm_utils->create_gep2(llvm_utils->get_type_from_ttype_t_util(x.m_ptr, p_type, module.get()), ptr, 1);
            ptr = llvm_utils->CreateLoad2(llvm_utils->get_type_from_ttype_t_util(x.m_ptr, p_type, module.get())->getPointerTo(), ptr);
        } else if(ASRUtils::is_character(*p_type)){ // String OR array of strings
            ptr = ASRUtils::is_array_of_strings(p_type) ? 
                llvm_utils->get_stringArray_data(p_type, ptr) :
                llvm_utils->get_string_data(ASRUtils::get_string_type(p_type), ptr);
        } else {
            llvm::Type* p_llvm_type = llvm_utils->get_type_from_ttype_t_util(x.m_ptr, p_type, module.get());
            ptr = llvm_utils->CreateLoad2(p_llvm_type, ptr);
        }
        if( ASRUtils::is_array(p_type) &&
            !ASRUtils::is_array_of_strings(p_type) &&
            ASRUtils::extract_physical_type(p_type) ==
            ASR::array_physical_typeType::DescriptorArray) {
            LCOMPILERS_ASSERT(ASR::is_a<ASR::Pointer_t>(*p_type));
            ptr = arr_descr->get_pointer_to_data(x.m_ptr,
                ASRUtils::type_get_past_allocatable_pointer(p_type), ptr, module.get());
            llvm::Type* array_inner_type = llvm_utils->get_type_from_ttype_t_util(x.m_ptr,
                ASRUtils::extract_type(p_type), module.get());
            ptr = llvm_utils->CreateLoad2(array_inner_type->getPointerTo(), ptr);
        }
#if LLVM_VERSION_MAJOR > 16
        ptr_type[ptr] = ptr_arr_type;
#endif
        ptr_loads = ptr_loads_copy;
        if( ASR::is_a<ASR::CPtr_t>(*ASRUtils::expr_type(x.m_ptr)) &&
            x.m_tgt && ASR::is_a<ASR::CPtr_t>(*ASRUtils::expr_type(x.m_tgt)) ) {
            int64_t ptr_loads_copy = ptr_loads;
            ptr_loads = 0;
            this->visit_expr_wrapper(x.m_tgt, true);
            ptr_loads = ptr_loads_copy;
            tmp = builder->CreateICmpEQ(
                builder->CreatePtrToInt(ptr, llvm_utils->getIntType(8, false)),
                builder->CreatePtrToInt(tmp, llvm_utils->getIntType(8, false)));
            return ;
        }
        llvm_utils->create_if_else(builder->CreateICmpEQ(
            builder->CreatePtrToInt(ptr, llvm_utils->getIntType(8, false)),
            llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), llvm::APInt(64, 0))),
        [&]() {
            builder->CreateStore(
                llvm::ConstantInt::get(llvm::Type::getInt1Ty(context), llvm::APInt(1, 0)),
                res);
        },
        [&]() {
            if (x.m_tgt) {
                int64_t ptr_loads_copy = ptr_loads;
                ptr_loads = 0;
                this->visit_expr_wrapper(x.m_tgt, true);
                ptr_loads = ptr_loads_copy;
                // ASR::Variable_t *t = EXPR2VAR(x.m_tgt);
                // uint32_t t_h = get_hash((ASR::asr_t*)t);
                // nptr = llvm_symtab[t_h];
                nptr = tmp;
                if( ASRUtils::is_array(ASRUtils::expr_type(x.m_tgt)) ) {
                    ASR::array_physical_typeType tgt_ptype = ASRUtils::extract_physical_type(
                        ASRUtils::expr_type(x.m_tgt));
                    llvm::Type* array_type = llvm_utils->get_type_from_ttype_t_util(x.m_tgt,
                        ASRUtils::expr_type(x.m_tgt), module.get());
                    if( tgt_ptype == ASR::array_physical_typeType::FixedSizeArray ) {
                        nptr = llvm_utils->create_gep2(array_type, nptr, 0);
                    } else if( tgt_ptype == ASR::array_physical_typeType::DescriptorArray ) {
                        llvm::Type* array_type = llvm_utils->get_type_from_ttype_t_util(x.m_tgt,
                            ASRUtils::expr_type(x.m_tgt), module.get());
                        llvm::Type *array_inner_type = llvm_utils->get_type_from_ttype_t_util(x.m_ptr,
                            ASRUtils::extract_type(p_type), module.get());
                        nptr = builder->CreateLoad(array_type, nptr);
                        nptr = llvm_utils->CreateLoad2(array_inner_type->getPointerTo(), arr_descr->get_pointer_to_data(nptr));
                    }
                }
                nptr = builder->CreatePtrToInt(nptr, llvm_utils->getIntType(8, false));
                ptr = builder->CreatePtrToInt(ptr, llvm_utils->getIntType(8, false));
                builder->CreateStore(builder->CreateICmpEQ(ptr, nptr), res);
            } else {
                llvm::Type* value_type = llvm_utils->get_type_from_ttype_t_util(x.m_ptr, p_type, module.get());
                nptr = llvm::ConstantPointerNull::get(static_cast<llvm::PointerType*>(value_type));
                nptr = builder->CreatePtrToInt(nptr, llvm_utils->getIntType(8, false));
                ptr = builder->CreatePtrToInt(ptr, llvm_utils->getIntType(8, false));
                builder->CreateStore(builder->CreateICmpNE(ptr, nptr), res);
            }
        });
        tmp = llvm_utils->CreateLoad(res);
    }

    void handle_array_section_association_to_pointer(const ASR::Associate_t& x) {
        ASR::ArraySection_t* array_section = ASR::down_cast<ASR::ArraySection_t>(x.m_value);
        ASR::ttype_t* value_array_type = ASRUtils::expr_type(array_section->m_v);
        bool is_parameter = ASRUtils::is_value_constant(array_section->m_v);

        int64_t ptr_loads_copy = ptr_loads;
        ptr_loads = 1 - !LLVM::is_llvm_pointer(*value_array_type);
        visit_expr_wrapper(array_section->m_v);
        llvm::Value* value_desc = tmp;
        if( ASR::is_a<ASR::StructInstanceMember_t>(*array_section->m_v) &&
            ASRUtils::extract_physical_type(value_array_type) !=
                ASR::array_physical_typeType::FixedSizeArray ) {
            value_desc = llvm_utils->CreateLoad(value_desc);
        }
#if LLVM_VERSION_MAJOR > 16
        ptr_type[value_desc] = llvm_utils->get_type_from_ttype_t_util(array_section->m_v,
            ASRUtils::type_get_past_allocatable_pointer(value_array_type),
            module.get());
#endif
        llvm::Type *value_el_type = llvm_utils->get_type_from_ttype_t_util(array_section->m_v,
              ASRUtils::extract_type(value_array_type), module.get());
        ptr_loads = 0;
        visit_expr(*x.m_target);
        llvm::Value* target_desc = tmp;
        ptr_loads = ptr_loads_copy;

        ASR::ttype_t* target_desc_type = ASRUtils::duplicate_type_with_empty_dims(al,
            ASRUtils::type_get_past_allocatable(
                ASRUtils::type_get_past_pointer(value_array_type)),
             ASR::array_physical_typeType::DescriptorArray, true);
        llvm::Type* target_type = llvm_utils->get_type_from_ttype_t_util(array_section->m_v, target_desc_type, module.get());
        llvm::AllocaInst *target = llvm_utils->CreateAlloca(
            target_type, nullptr, "array_section_descriptor");
        if( ASRUtils::is_character(*expr_type(x.m_target))){
            llvm::Value* str_desc = llvm_utils->create_string_descriptor("array_section_string_desc");
            builder->CreateStore(str_desc, arr_descr->get_pointer_to_data(target));
        }
#if LLVM_VERSION_MAJOR > 16
        ptr_type[target] = target_type;
#endif
        int value_rank = array_section->n_args, target_rank = 0;
        Vec<llvm::Value*> lbs; lbs.reserve(al, value_rank);
        Vec<llvm::Value*> ubs; ubs.reserve(al, value_rank);
        Vec<llvm::Value*> ds; ds.reserve(al, value_rank);
        Vec<llvm::Value*> non_sliced_indices; non_sliced_indices.reserve(al, value_rank);
        for( int i = 0; i < value_rank; i++ ) {
            lbs.p[i] = nullptr; ubs.p[i] = nullptr; ds.p[i] = nullptr;
            non_sliced_indices.p[i] = nullptr;
            if( array_section->m_args[i].m_step != nullptr ) {
                visit_expr_wrapper(array_section->m_args[i].m_left, true);
                lbs.p[i] = tmp;
                visit_expr_wrapper(array_section->m_args[i].m_right, true);
                ubs.p[i] = tmp;
                visit_expr_wrapper(array_section->m_args[i].m_step, true); 
                ds.p[i] = tmp;
                target_rank++;
            } else {
                visit_expr_wrapper(array_section->m_args[i].m_right, true);
                non_sliced_indices.p[i] = tmp;
            }
        }
        LCOMPILERS_ASSERT(target_rank > 0);
        llvm::Value* target_dim_des_ptr = arr_descr->get_pointer_to_dimension_descriptor_array(target, false);
        llvm::Value* target_dim_des_val = llvm_utils->CreateAlloca(arr_descr->get_dimension_descriptor_type(false),
            llvm::ConstantInt::get(llvm_utils->getIntType(4), llvm::APInt(32, target_rank)));
        builder->CreateStore(target_dim_des_val, target_dim_des_ptr);
        ASR::ttype_t* array_type = ASRUtils::expr_type(array_section->m_v);
        int dims = ASRUtils::extract_n_dims_from_ttype(array_type);
        ASR::array_physical_typeType arr_physical_type = ASRUtils::extract_physical_type(array_type);
        if( arr_physical_type == ASR::array_physical_typeType::PointerToDataArray ||
            arr_physical_type == ASR::array_physical_typeType::FixedSizeArray ||
            arr_physical_type == ASR::array_physical_typeType::StringArraySinglePointer) {
            if( (arr_physical_type == ASR::array_physical_typeType::FixedSizeArray ||
                arr_physical_type == ASR::array_physical_typeType::StringArraySinglePointer) &&
                !(is_parameter && dims == 1) ) {
                llvm::Type *val_type = llvm_utils->get_type_from_ttype_t_util(array_section->m_v,
                    ASRUtils::type_get_past_allocatable(
                    ASRUtils::type_get_past_pointer(value_array_type)),
                    module.get());
                value_desc = llvm_utils->create_gep2(val_type, value_desc, 0);
            }
            ASR::dimension_t* m_dims = nullptr;
            // Fill in m_dims:
            [[maybe_unused]] int array_value_rank = ASRUtils::extract_dimensions_from_ttype(array_type, m_dims);
            LCOMPILERS_ASSERT(array_value_rank == value_rank);
            Vec<llvm::Value*> llvm_diminfo;
            llvm_diminfo.reserve(al, value_rank * 2);
            for( int i = 0; i < value_rank; i++ ) {
                visit_expr_wrapper(m_dims[i].m_start, true);
                llvm_diminfo.push_back(al, tmp);
                visit_expr_wrapper(m_dims[i].m_length, true);
                llvm_diminfo.push_back(al, tmp);
            }
            arr_descr->fill_descriptor_for_array_section_data_only(value_desc, value_el_type, expr_type(x.m_value),
                target, expr_type(x.m_target),
                lbs.p, ubs.p, ds.p, non_sliced_indices.p,
                llvm_diminfo.p, value_rank, target_rank);
        } else {
            arr_descr->fill_descriptor_for_array_section(value_desc, value_el_type, expr_type(x.m_value),
                target, expr_type(x.m_target),
                lbs.p, ubs.p, ds.p, non_sliced_indices.p,
                array_section->n_args, target_rank);
        }
        builder->CreateStore(target, target_desc);
    }

    void visit_Associate(const ASR::Associate_t& x) {
        if( ASR::is_a<ASR::ArraySection_t>(*x.m_value) ) {
            handle_array_section_association_to_pointer(x);
        } else {
            int64_t ptr_loads_copy = ptr_loads;
            ptr_loads = 0;
            visit_expr(*x.m_target);
            llvm::Value* llvm_target = tmp;
            visit_expr(*x.m_value);
            llvm::Value* llvm_value = tmp;
            ptr_loads = ptr_loads_copy;
            ASR::ttype_t* target_type = ASRUtils::expr_type(x.m_target);
            ASR::ttype_t* value_type = ASRUtils::expr_type(x.m_value);
            ASR::dimension_t* m_dims = nullptr;
            int n_dims = ASRUtils::extract_dimensions_from_ttype(target_type, m_dims);
            ASR::ttype_t *type = ASRUtils::get_contained_type(target_type);
            type = ASRUtils::type_get_past_allocatable(type);
            [[maybe_unused]] bool is_target_class = ASRUtils::is_class_type(
                ASRUtils::type_get_past_pointer(target_type));
            [[maybe_unused]] bool is_value_class = ASRUtils::is_class_type(
                ASRUtils::type_get_past_pointer(
                    ASRUtils::type_get_past_allocatable(value_type)));
            llvm::Type *i64 = llvm::Type::getInt64Ty(context);
            if (ASR::is_a<ASR::PointerNullConstant_t>(*x.m_value)) {
                if(ASRUtils::is_array(target_type) ){ // Fetch data ptr
                    LCOMPILERS_ASSERT(ASRUtils::extract_physical_type(target_type) ==
                        ASR::array_physical_typeType::DescriptorArray);
                    llvm_target = llvm_utils->CreateLoad(llvm_target);
                    llvm_target = arr_descr->get_pointer_to_data(llvm_target);
                }
                builder->CreateStore(llvm_value, llvm_target);
            } else if(ASRUtils::is_string_only(value_type)){ // Maybe we can handle this case in a string api function
                LCOMPILERS_ASSERT(ASRUtils::is_string_only(target_type));
                llvm::Value* target_ptr = llvm_utils->get_string_data(
                    ASRUtils::get_string_type(target_type), llvm_target, true); //i8**
                llvm::Value* value_ptr = llvm_utils->get_string_data(
                    ASRUtils::get_string_type(target_type), llvm_value); // i8*
                builder->CreateStore(value_ptr, target_ptr);
                switch (ASRUtils::get_string_type(target_type)->m_physical_type)
                {
                    case ASR::DescriptorString:{
                        llvm::Value* target_length = llvm_utils->get_string_length(
                            ASRUtils::get_string_type(target_type), llvm_target, true); // i64*
                        llvm::Value* value_length = llvm_utils->get_string_length(
                            ASRUtils::get_string_type(value_type), llvm_value); // i64
                        builder->CreateStore(value_length, target_length);
                        break;
                    }
                    default:
                        throw LCompilersException("Unhandled String Physical Type");
                }
            } else if (ASR::is_a<ASR::StructInstanceMember_t>(*x.m_value) &&
                        ASR::is_a<ASR::FunctionType_t>(*value_type)) {
                llvm_value = llvm_utils->CreateLoad(llvm_value);
                builder->CreateStore(llvm_value, llvm_target);
            } else if (is_target_class && !is_value_class) {
                llvm::Type* llvm_target_type = llvm_utils->get_type_from_ttype_t_util(x.m_target, target_type, module.get());
                llvm::Value* vtab_address_ptr = llvm_utils->create_gep2(llvm_target_type, llvm_target, 0);
                llvm_target = llvm_utils->create_gep2(llvm_target_type, llvm_target, 1);
                ASR::symbol_t* struct_sym = ASRUtils::symbol_get_past_external(ASRUtils::get_struct_sym_from_struct_expr(x.m_value));
                if (type2vtab.find(struct_sym) == type2vtab.end() ||
                    type2vtab[struct_sym].find(current_scope) == type2vtab[struct_sym].end()) {
                    create_vtab_for_struct_type(struct_sym, current_scope);
                }
                llvm::Value* vtab_obj = type2vtab[struct_sym][current_scope];
                llvm::Type* vtab_struct_type = llvm_utils->getStructType(
                    ASR::down_cast<ASR::Struct_t>(struct_sym), module.get(), false);
                llvm::Value* vtab_obj_casted = builder->CreateBitCast(vtab_obj, vtab_struct_type->getPointerTo());
                llvm::Value* gep = llvm_utils->create_gep2(vtab_struct_type, vtab_obj_casted, 0);
                llvm::Value* struct_type_hash = llvm_utils->CreateLoad2(i64, gep);
                builder->CreateStore(struct_type_hash, vtab_address_ptr);

                ASR::Struct_t* struct_type_t = ASR::down_cast<ASR::Struct_t>(
                    ASRUtils::symbol_get_past_external(ASRUtils::get_struct_sym_from_struct_expr(x.m_target)));
                llvm_value = builder->CreateBitCast(
                    llvm_value, llvm_utils->getStructType(struct_type_t, module.get(), true));
                builder->CreateStore(llvm_value, llvm_target);
            } else if (!is_target_class && is_value_class) {
                llvm::Type* llvm_value_type = llvm_utils->get_type_from_ttype_t_util(x.m_value, value_type, module.get());
                llvm::Value* val_data_ptr = llvm_utils->create_gep2(llvm_value_type, llvm_value, 1);
              
                ASR::Struct_t* struct_type_t = ASR::down_cast<ASR::Struct_t>(
                    ASRUtils::symbol_get_past_external(ASRUtils::symbol_get_past_external(ASRUtils::get_struct_sym_from_struct_expr(x.m_target))));
                [[maybe_unused]] ASR::Struct_t* class_type_t = ASR::down_cast<ASR::Struct_t>(
                    ASRUtils::symbol_get_past_external(ASRUtils::symbol_get_past_external(ASRUtils::get_struct_sym_from_struct_expr(x.m_value))));
                LCOMPILERS_ASSERT(ASRUtils::is_derived_type_similar(struct_type_t, class_type_t));
                llvm::Type* struct_type = llvm_utils->getStructType(struct_type_t, module.get(), true);
                llvm::Value* casted_val_ptr = builder->CreateBitCast(val_data_ptr, struct_type->getPointerTo());

                llvm::Value* struct_value = builder->CreateLoad(struct_type, casted_val_ptr);
                builder->CreateStore(struct_value, llvm_target);
            } else if( is_target_class && is_value_class ) {
                std::string value_struct_t_name = "";
                ASR::Struct_t* value_struct_t = ASR::down_cast<ASR::Struct_t>(
                        ASRUtils::symbol_get_past_external(ASRUtils::symbol_get_past_external(ASRUtils::get_struct_sym_from_struct_expr(x.m_value))));
                value_struct_t_name = value_struct_t->m_name;
                LCOMPILERS_ASSERT(ASRUtils::symbol_get_past_external(ASRUtils::get_struct_sym_from_struct_expr(x.m_target))
                                  == ASRUtils::symbol_get_past_external(ASRUtils::get_struct_sym_from_struct_expr(x.m_value)));
                llvm::Type* value_llvm_type = llvm_utils->get_type_from_ttype_t_util(x.m_value,
                    ASRUtils::type_get_past_allocatable_pointer(value_type), module.get());
                llvm::Type* ptr_type = llvm_utils->getStructType(
                    ASR::down_cast<ASR::Struct_t>(
                        ASRUtils::symbol_get_past_external(ASRUtils::get_struct_sym_from_struct_expr(x.m_value))), module.get(), true);
                llvm::Value* gep = llvm_utils->create_gep2(value_llvm_type, llvm_value, 0);
                llvm::Value* value_vtabid = llvm_utils->CreateLoad2(i64, gep);
                llvm::Value* value_class = llvm_utils->CreateLoad2(ptr_type, llvm_utils->create_gep2(value_llvm_type, llvm_value, 1));
                builder->CreateStore(value_vtabid, llvm_utils->create_gep2(value_llvm_type, llvm_target, 0));

                if ( value_struct_t_name == "~unlimited_polymorphic_type" ) {
                    // we need to cast `value_class` to `void*`
                    llvm::Type* void_ptr_type = llvm::Type::getVoidTy(context)->getPointerTo();
                    value_class = builder->CreateBitCast(value_class, void_ptr_type);
                }
                builder->CreateStore(value_class, llvm_utils->create_gep2(value_llvm_type, llvm_target, 1));
            } else if (ASR::is_a<ASR::Pointer_t>(*value_type) &&
                       ASR::is_a<ASR::Pointer_t>(*target_type) &&
                       ASR::is_a<ASR::FunctionCall_t>(*x.m_value)) {
                builder->CreateStore(llvm_value, llvm_target);
            } else {
                bool is_value_data_only_array = (ASRUtils::is_array(value_type) && (
                      ASRUtils::extract_physical_type(value_type) == ASR::array_physical_typeType::PointerToDataArray ||
                      ASRUtils::extract_physical_type(value_type) == ASR::array_physical_typeType::FixedSizeArray ||
                      ASRUtils::extract_physical_type(value_type) == ASR::array_physical_typeType::DescriptorArray));
                if( LLVM::is_llvm_pointer(*value_type) ) {
                    llvm_value = llvm_utils->CreateLoad(llvm_value);
                }
                if( is_value_data_only_array ) { // This needs a refactor to handle
                    ASR::ttype_t* target_type_ = ASRUtils::type_get_past_pointer(target_type);
                    switch( ASRUtils::extract_physical_type(target_type_) ) {
                        case ASR::array_physical_typeType::DescriptorArray: {
                            if(ASRUtils::extract_physical_type(value_type) == ASR::array_physical_typeType::DescriptorArray){
                                // Declare llvm type for (Array descriptor, Dimension Descriptor)
                                llvm::Type* const array_desc_type = llvm_utils->arr_api->
                                    get_array_type(x.m_value, ASRUtils::type_get_past_allocatable_pointer(value_type),
                                        llvm_utils->get_el_type(x.m_value, ASRUtils::extract_type(value_type), module.get()), false);
                                LCOMPILERS_ASSERT(array_desc_type->isStructTy());
                                llvm::Type* const dim_desc_type = llvm_utils->arr_api->get_dimension_descriptor_type(false);
                                llvm::Value* llvm_target_ = builder->CreateLoad(array_desc_type->getPointerTo(), llvm_target);
                                // Store exact data pointer
                                llvm::Value* value_data_ptr = llvm_utils->create_gep2(array_desc_type, llvm_value, 0); // Pointer to data of the RHS array.
                                llvm::Value* target_data_ptr = llvm_utils->create_gep2(array_desc_type, llvm_target_, 0); // Pointer to data of the LHS array.
                                builder->CreateStore(builder->CreateLoad(
                                    llvm_utils->get_el_type(x.m_value,
                                        ASRUtils::extract_type(value_type), module.get())->getPointerTo(), value_data_ptr),
                                    target_data_ptr);
                                // Deep Copy dimension descriptor
                                llvm::Value* value_dim_ptr = builder->CreateLoad(dim_desc_type->getPointerTo(),
                                                                 llvm_utils->create_gep2(array_desc_type, llvm_value, 2)); // Pointer to dimension descriptor of the RHS array.
                                llvm::Value* target_dim_ptr = builder->CreateLoad(dim_desc_type->getPointerTo(),
                                                            llvm_utils->create_gep2(array_desc_type, llvm_target_, 2)); // Pointer to dimension descriptor of the LHS array.
                                llvm::DataLayout data_layout(module->getDataLayout());
                                int dim_desc_size = (int)data_layout.getTypeAllocSize(dim_desc_type);
                                builder->CreateMemCpy(target_dim_ptr, llvm::MaybeAlign(8), value_dim_ptr, llvm::MaybeAlign(8), dim_desc_size*n_dims);
                                // Copy offset
                                llvm::Value* value_offset = llvm_utils->create_gep2(array_desc_type, llvm_value, 1); // Pointer to offset of the RHS array.
                                llvm::Value* target_offset = llvm_utils->create_gep2(array_desc_type, llvm_target_, 1); // Pointer to offset of the LHS array.
                                builder->CreateStore(builder->CreateLoad(llvm::Type::getInt32Ty(context),value_offset), target_offset);
                                // Other fields of the array descriptor should be already set.
                                return;
                            }else if( ASRUtils::extract_physical_type(value_type) == ASR::array_physical_typeType::FixedSizeArray ) {
                                llvm::Type* llvm_type = llvm_utils->get_type_from_ttype_t_util(x.m_value,
                                    value_type, module.get());
                                llvm_value = llvm_utils->create_gep2(llvm_type, llvm_value, 0);
                            }
                            llvm::Type* llvm_target_type = llvm_utils->get_type_from_ttype_t_util(x.m_target, target_type_, module.get());
                            llvm::Value* llvm_target_ = llvm_utils->CreateAlloca(*builder, llvm_target_type);
                            ASR::dimension_t* m_dims = nullptr;
                            size_t n_dims = ASRUtils::extract_dimensions_from_ttype(value_type, m_dims);
                            ASR::ttype_t* data_type = ASRUtils::duplicate_type_without_dims(
                                                        al, target_type_, target_type_->base.loc);
                            llvm::Type* llvm_data_type = llvm_utils->get_type_from_ttype_t_util(x.m_target, data_type, module.get());
                            fill_array_details(llvm_target_, llvm_data_type, m_dims, n_dims, false, false);
                            builder->CreateStore(llvm_value, arr_descr->get_pointer_to_data(llvm_target_));
                            llvm_value = llvm_target_;
                            break;
                        }
                        case ASR::array_physical_typeType::FixedSizeArray: {
                            llvm_value = llvm_utils->CreateLoad(llvm_value);
                            break;
                        }
                        case ASR::array_physical_typeType::PointerToDataArray: {
                            break;
                        }
                        default: {
                            LCOMPILERS_ASSERT(false);
                        }
                    }
                }
                builder->CreateStore(llvm_value, llvm_target);
            }
        }
    }

    void handle_StringSection_Assignment(ASR::expr_t *target, ASR::expr_t *value) {
        // Handles the case when LHS of assignment is string.
        std::string runtime_func_name = "_lfortran_str_slice_assign";
        llvm::Function *fn = module->getFunction(runtime_func_name);
        if (!fn) {
            llvm::FunctionType *function_type = llvm::FunctionType::get(
                    character_type, {
                        character_type, llvm::Type::getInt64Ty(context),
                        character_type, llvm::Type::getInt64Ty(context),
                        llvm::Type::getInt32Ty(context),
                        llvm::Type::getInt32Ty(context), llvm::Type::getInt32Ty(context),
                        llvm::Type::getInt1Ty(context), llvm::Type::getInt1Ty(context)
                    }, false);
            fn = llvm::Function::Create(function_type,
                    llvm::Function::ExternalLinkage, runtime_func_name, *module);
        }
        ASR::StringSection_t *ss = ASR::down_cast<ASR::StringSection_t>(target);
        llvm::Value *lp, *rp;
        llvm::Value *str, *idx1, *idx2, *step, *str_val;
        visit_expr_load_wrapper(ss->m_arg, 0,true);
        str = tmp;
        visit_expr_load_wrapper(value, 0,true);
        str_val = tmp;
        if (ss->m_start) {
            this->visit_expr_wrapper(ss->m_start, true);
            idx1 = tmp;
            lp = llvm::ConstantInt::get(context,
                llvm::APInt(1, 1));
        } else {
            lp = llvm::ConstantInt::get(context,
                llvm::APInt(1, 0));
            idx1 = llvm::Constant::getNullValue(llvm::Type::getInt32Ty(context));
        }
        if (ss->m_end) {
            this->visit_expr_wrapper(ss->m_end, true);
            idx2 = tmp;
            rp = llvm::ConstantInt::get(context,
                llvm::APInt(1, 1));
        } else {
            rp = llvm::ConstantInt::get(context,
                llvm::APInt(1, 0));
            idx2 = llvm::Constant::getNullValue(llvm::Type::getInt32Ty(context));
        }
        if (ss->m_step) {
            this->visit_expr_wrapper(ss->m_step, true);
            step = tmp;
        } else {
            step = llvm::ConstantInt::get(context,
                llvm::APInt(32, 0));
        }
        llvm::Value* str_data, *str_len;
        std::tie(str_data, str_len) = llvm_utils->get_string_length_data(ASRUtils::get_string_type(ss->m_arg), str);

        llvm::Value* str_val_data, *str_val_len;
        std::tie(str_val_data, str_val_len) = llvm_utils->get_string_length_data(ASRUtils::get_string_type(value), str_val);

        tmp = builder->CreateCall(fn, {str_data, str_len, str_val_data, str_val_len, idx1, idx2, step, lp, rp});

        strings_to_be_deallocated.push_back(al, tmp); // char* returing from this call is dead after making the next str_copy call.

        tmp = builder->CreateMemCpy(
            llvm_utils->get_string_data(ASRUtils::get_string_type(ss->m_arg), str),llvm::MaybeAlign(8),
            tmp, llvm::MaybeAlign(8),
            llvm_utils->get_string_length(ASRUtils::get_string_type(ss->m_arg), str));
    }

    void visit_OverloadedStringConcat(const ASR::OverloadedStringConcat_t &x) {
        LCOMPILERS_ASSERT(x.m_overloaded != nullptr)
        this->visit_expr(*x.m_overloaded);
    }

    void visit_Assignment(const ASR::Assignment_t &x) {
        if (compiler_options.emit_debug_info) debug_emit_loc(x);
        if( x.m_overloaded ) {
            this->visit_stmt(*x.m_overloaded);
            return ;
        }

        ASR::ttype_t* asr_target_type = ASRUtils::expr_type(x.m_target);
        ASR::ttype_t* asr_value_type = ASRUtils::expr_type(x.m_value);
        bool is_target_list = ASR::is_a<ASR::List_t>(*asr_target_type);
        bool is_value_list = ASR::is_a<ASR::List_t>(*asr_value_type);
        bool is_target_tuple = ASR::is_a<ASR::Tuple_t>(*asr_target_type);
        bool is_value_tuple = ASR::is_a<ASR::Tuple_t>(*asr_value_type);
        bool is_target_dict = ASR::is_a<ASR::Dict_t>(*asr_target_type);
        bool is_value_dict = ASR::is_a<ASR::Dict_t>(*asr_value_type);
        bool is_target_set = ASR::is_a<ASR::Set_t>(*asr_target_type);
        bool is_value_set = ASR::is_a<ASR::Set_t>(*asr_value_type);
        bool is_target_struct = ASR::is_a<ASR::StructType_t>(
            *ASRUtils::type_get_past_allocatable(asr_target_type)) &&
            !ASRUtils::is_class_type(ASRUtils::type_get_past_allocatable(asr_target_type));
        bool is_value_struct = ASR::is_a<ASR::StructType_t>(
            *ASRUtils::type_get_past_allocatable(asr_value_type)) &&
             !ASRUtils::is_class_type(ASRUtils::type_get_past_allocatable(asr_value_type));
        // a class variable is always either an allocatable, a pointer, or a dummy argument of a procedure
        bool is_target_class = ASR::is_a<ASR::StructType_t>(
                                   *ASRUtils::type_get_past_allocatable_pointer(asr_target_type))
                               && ASRUtils::is_class_type(
                                   ASRUtils::type_get_past_allocatable_pointer(asr_target_type));
        bool is_value_class = ASR::is_a<ASR::StructType_t>(
                                  *ASRUtils::type_get_past_allocatable_pointer(asr_value_type))
                              && ASRUtils::is_class_type(
                                  ASRUtils::type_get_past_allocatable_pointer(asr_value_type));
        bool is_value_list_to_array = (ASR::is_a<ASR::Cast_t>(*x.m_value) &&
            ASR::down_cast<ASR::Cast_t>(x.m_value)->m_kind == ASR::cast_kindType::ListToArray);

        llvm::Type* asr_target_llvm_type = llvm_utils->get_type_from_ttype_t_util(x.m_target, asr_target_type, module.get());
        if (ASR::is_a<ASR::StringSection_t>(*x.m_target)) {
            handle_StringSection_Assignment(x.m_target, x.m_value);
            if (tmp == strings_to_be_deallocated.back()) {
                strings_to_be_deallocated.erase(strings_to_be_deallocated.back());
            }
            return;
        }
        if( is_target_list && is_value_list ) {
            int64_t ptr_loads_copy = ptr_loads;
            ptr_loads = 0;
            this->visit_expr(*x.m_target);
            llvm::Value* target_list = tmp;
            this->visit_expr(*x.m_value);
            llvm::Value* value_list = tmp;
            ptr_loads = ptr_loads_copy;
            ASR::List_t* value_asr_list = ASR::down_cast<ASR::List_t>(
                                            ASRUtils::expr_type(x.m_value));
            std::string value_type_code = ASRUtils::get_type_code(value_asr_list->m_type);
            list_api->list_deepcopy(x.m_value, value_list, target_list,
                                    value_asr_list, module.get(),
                                    name2memidx);
            return ;
        } else if( is_target_tuple && is_value_tuple ) {
            int64_t ptr_loads_copy = ptr_loads;
            if( ASR::is_a<ASR::TupleConstant_t>(*x.m_target) &&
                !ASR::is_a<ASR::TupleConstant_t>(*x.m_value) ) {
                ptr_loads = 0;
                this->visit_expr(*x.m_value);
                llvm::Value* value_tuple = tmp;
                ASR::TupleConstant_t* const_tuple = ASR::down_cast<ASR::TupleConstant_t>(x.m_target);
                ASR::Tuple_t* tuple_type = ASR::down_cast<ASR::Tuple_t>(const_tuple->m_type);
                for( size_t i = 0; i < const_tuple->n_elements; i++ ) {
                    ptr_loads = 0;
                    visit_expr(*const_tuple->m_elements[i]);
                    llvm::Value* target_ptr = tmp;
                    llvm::Value* item = tuple_api->read_item(value_tuple, tuple_type, i, false);
                    builder->CreateStore(item, target_ptr);
                }
                ptr_loads = ptr_loads_copy;
            } else if( ASR::is_a<ASR::TupleConstant_t>(*x.m_target) &&
                       ASR::is_a<ASR::TupleConstant_t>(*x.m_value) ) {
                ASR::TupleConstant_t* asr_value_tuple = ASR::down_cast<ASR::TupleConstant_t>(x.m_value);
                ASR::TupleConstant_t* asr_target_tuple = ASR::down_cast<ASR::TupleConstant_t>(x.m_target);
                Vec<llvm::Value*> src_deepcopies;
                src_deepcopies.reserve(al, asr_value_tuple->n_elements);
                for( size_t i = 0; i < asr_value_tuple->n_elements; i++ ) {
                    ASR::ttype_t* asr_tuple_i_type = ASRUtils::expr_type(asr_value_tuple->m_elements[i]);
                    llvm::Type* llvm_tuple_i_type = llvm_utils->get_type_from_ttype_t_util(asr_value_tuple->m_elements[i], asr_tuple_i_type, module.get());
                    llvm::Value* llvm_tuple_i = llvm_utils->CreateAlloca(*builder, llvm_tuple_i_type);
                    ptr_loads = !LLVM::is_llvm_struct(asr_tuple_i_type);
                    visit_expr(*asr_value_tuple->m_elements[i]);
                    llvm_utils->deepcopy(asr_value_tuple->m_elements[i], tmp, llvm_tuple_i, expr_type(asr_target_tuple->m_elements[i]), asr_tuple_i_type, module.get(), name2memidx);
                    src_deepcopies.push_back(al, llvm_tuple_i);
                }
                for( size_t i = 0; i < asr_target_tuple->n_elements; i++ ) {
                    ptr_loads = 0;
                    visit_expr(*asr_target_tuple->m_elements[i]);
                    LLVM::CreateStore(*builder,
                        llvm_utils->CreateLoad(src_deepcopies[i]),
                        tmp
                    );
                }
                ptr_loads = ptr_loads_copy;
            } else {
                ptr_loads = 0;
                this->visit_expr(*x.m_value);
                llvm::Value* value_tuple = tmp;
                this->visit_expr(*x.m_target);
                llvm::Value* target_tuple = tmp;
                ptr_loads = ptr_loads_copy;
                ASR::Tuple_t* value_tuple_type = ASR::down_cast<ASR::Tuple_t>(asr_value_type);
                std::string type_code = ASRUtils::get_type_code(value_tuple_type->m_type,
                                                                value_tuple_type->n_type);
                tuple_api->tuple_deepcopy(x.m_value, value_tuple, target_tuple,
                                          value_tuple_type, module.get(),
                                          name2memidx);
            }
            return ;
        } else if( is_target_dict && is_value_dict ) {
            int64_t ptr_loads_copy = ptr_loads;
            ptr_loads = 0;
            this->visit_expr(*x.m_value);
            llvm::Value* value_dict = tmp;
            this->visit_expr(*x.m_target);
            llvm::Value* target_dict = tmp;
            ptr_loads = ptr_loads_copy;
            ASR::Dict_t* value_dict_type = ASR::down_cast<ASR::Dict_t>(asr_value_type);
            llvm_utils->set_dict_api(value_dict_type);

            std::string key_type_code = ASRUtils::get_type_code(value_dict_type->m_key_type);
            std::string value_type_code = ASRUtils::get_type_code(value_dict_type->m_value_type);
            llvm_utils->dict_api->dict_deepcopy(nullptr, value_dict, target_dict,
                                    value_dict_type, module.get(), name2memidx);
            return ;
        } else if( is_target_set && is_value_set ) {
            int64_t ptr_loads_copy = ptr_loads;
            ptr_loads = 0;
            this->visit_expr(*x.m_value);
            llvm::Value* value_set = tmp;
            this->visit_expr(*x.m_target);
            llvm::Value* target_set = tmp;
            ptr_loads = ptr_loads_copy;
            ASR::Set_t* value_set_type = ASR::down_cast<ASR::Set_t>(asr_value_type);
            llvm_utils->set_set_api(value_set_type);
            llvm_utils->set_api->set_deepcopy(x.m_value, value_set, target_set,
                                    value_set_type, module.get(), name2memidx);
            return ;
        } else if( is_target_struct && is_value_struct ) {
            int64_t ptr_loads_copy = ptr_loads;
            ptr_loads = 0;
            this->visit_expr(*x.m_value);
            llvm::Value* value_struct = tmp;
            bool is_assignment_target_copy = is_assignment_target;
            is_assignment_target = true;
            this->visit_expr(*x.m_target);
            is_assignment_target = is_assignment_target_copy;
            llvm::Value* target_struct = tmp;
            ptr_loads = ptr_loads_copy;
            if (ASRUtils::is_allocatable(asr_target_type)) {
                llvm::Type* tar_type = llvm_utils->get_type_from_ttype_t_util(x.m_target, asr_target_type, module.get());
                target_struct = llvm_utils->CreateLoad2(tar_type, target_struct);
            } 
            if (ASRUtils::is_allocatable(asr_value_type)) {
                llvm::Type* val_type = llvm_utils->get_type_from_ttype_t_util(x.m_value, asr_value_type, module.get());
                value_struct = llvm_utils->CreateLoad2(val_type, value_struct);
            }
            llvm_utils->deepcopy(x.m_value, value_struct, target_struct,
                asr_value_type, ASRUtils::type_get_past_allocatable(asr_target_type), module.get(), name2memidx);
            return ;
        } else if (is_target_class && is_value_class) {
            int64_t ptr_loads_copy = ptr_loads;
            ptr_loads = 0;
            this->visit_expr(*x.m_value);
            llvm::Value* value_struct = tmp;
            bool is_assignment_target_copy = is_assignment_target;
            is_assignment_target = true;
            this->visit_expr(*x.m_target);
            is_assignment_target = is_assignment_target_copy;
            llvm::Value* target_struct = tmp;
            ptr_loads = ptr_loads_copy;
            llvm::Type *i64 = llvm::Type::getInt64Ty(context);

            // deepcopy the class hash
            llvm::Type* value_llvm_type = llvm_utils->get_type_from_ttype_t_util(x.m_value, asr_value_type, module.get());
            llvm::Type* target_llvm_type = llvm_utils->get_type_from_ttype_t_util(x.m_target, asr_target_type, module.get());

            llvm::Value* value_class_hash = llvm_utils->create_gep2(value_llvm_type, value_struct, 0);
            value_class_hash = llvm_utils->CreateLoad2(i64, value_class_hash);

            llvm::Value* target_class_hash_ptr = llvm_utils->create_gep2(target_llvm_type, target_struct, 0);

            builder->CreateStore(value_class_hash, target_class_hash_ptr);

            // deepcopy the class ptr
            ASR::ttype_t* wrapped_value_struct_type = ASRUtils::make_StructType_t_util(al, asr_value_type->base.loc,
                            ASRUtils::symbol_get_past_external(ASRUtils::get_struct_sym_from_struct_expr(x.m_value)));
            llvm::Type* wrapper_value_llvm_type = llvm_utils->get_type_from_ttype_t_util(x.m_value, wrapped_value_struct_type, module.get());

            ASR::ttype_t* wrapped_target_struct_type = ASRUtils::make_StructType_t_util(al, asr_target_type->base.loc,
                            ASRUtils::symbol_get_past_external(ASRUtils::get_struct_sym_from_struct_expr(x.m_target)));
            llvm::Type* wrapper_target_llvm_type = llvm_utils->get_type_from_ttype_t_util(x.m_target, wrapped_target_struct_type, module.get());


            llvm::Value* value_class_ptr = llvm_utils->create_gep2(value_llvm_type, value_struct, 1);
            value_class_ptr = llvm_utils->CreateLoad2(wrapper_value_llvm_type->getPointerTo(), value_class_ptr);
            // bitcast to the correct type
            value_class_ptr = builder->CreateBitCast(value_class_ptr, wrapper_target_llvm_type->getPointerTo());
            value_class_ptr = llvm_utils->CreateLoad2(wrapper_target_llvm_type, value_class_ptr);

            llvm::Value* target_class_ptr = llvm_utils->create_gep2(target_llvm_type, target_struct, 1);
            target_class_ptr = llvm_utils->CreateLoad2(wrapper_target_llvm_type->getPointerTo(), target_class_ptr);

            builder->CreateStore(value_class_ptr, target_class_ptr);
            return;
        } else if (ASR::is_a<ASR::Allocatable_t>(*asr_target_type) &&
                   ASRUtils::is_class_type(ASRUtils::type_get_past_allocatable(asr_target_type)) &&
                   is_value_struct) {
            if (x.m_realloc_lhs) {
                check_and_allocate(x.m_target, x.m_value, asr_value_type);
            }
            int64_t ptr_loads_copy = ptr_loads;
            ptr_loads = 0;
            this->visit_expr(*x.m_value);
            llvm::Value* value_struct = tmp;

            bool is_assignment_target_copy = is_assignment_target;
            is_assignment_target = true;
            this->visit_expr(*x.m_target);
            is_assignment_target = is_assignment_target_copy;

            // Get and bitcast the wrapped struct in classtype
            ASR::ttype_t* wrapped_struct_type = ASRUtils::make_StructType_t_util(al, asr_target_type->base.loc,
                            ASRUtils::symbol_get_past_external(ASRUtils::get_struct_sym_from_struct_expr(x.m_target)));
            llvm::Type* wrapper_struct_llvm_type = llvm_utils->get_type_from_ttype_t_util(x.m_target, wrapped_struct_type, module.get())->getPointerTo();
            tmp = llvm_utils->create_gep2(asr_target_llvm_type, tmp, 1);
            tmp = llvm_utils->CreateLoad2(wrapper_struct_llvm_type, tmp);
            tmp = builder->CreateBitCast(tmp, value_struct->getType());
            ptr_loads = ptr_loads_copy;
            llvm::Value* target_struct = tmp;

            llvm_utils->deepcopy(x.m_value, value_struct, target_struct,
                asr_value_type, asr_value_type, module.get(), name2memidx);
            return;
        }

        if( ASR::is_a<ASR::Pointer_t>(*ASRUtils::expr_type(x.m_target)) &&
            ASR::is_a<ASR::GetPointer_t>(*x.m_value) ) {
            ASR::Variable_t *asr_target = EXPR2VAR(x.m_target);
            ASR::GetPointer_t* get_ptr = ASR::down_cast<ASR::GetPointer_t>(x.m_value);
            uint32_t target_h = get_hash((ASR::asr_t*)asr_target);
            int ptr_loads_copy = ptr_loads;
            ptr_loads = 2 - LLVM::is_llvm_pointer(*ASRUtils::expr_type(get_ptr->m_arg));
            visit_expr_wrapper(get_ptr->m_arg, true);
            ptr_loads = ptr_loads_copy;
            if( ASRUtils::is_array(ASRUtils::expr_type(get_ptr->m_arg)) &&
                ASRUtils::extract_physical_type(ASRUtils::expr_type(get_ptr->m_arg)) !=
                ASR::array_physical_typeType::DescriptorArray) {
                visit_ArrayPhysicalCastUtil(
                    tmp, get_ptr->m_arg, ASRUtils::type_get_past_pointer(
                        ASRUtils::type_get_past_allocatable(get_ptr->m_type)),
                        ASRUtils::expr_type(get_ptr->m_arg),
                    ASRUtils::extract_physical_type(ASRUtils::expr_type(get_ptr->m_arg)),
                    ASR::array_physical_typeType::DescriptorArray);
            }
            builder->CreateStore(tmp, llvm_symtab[target_h]);
            return ;
        }

        // When assigning to a StructInstanceMember, whose instance is allocatable
        // Check if the underlying struct instance is allocated, if not allocate
        if (x.m_realloc_lhs &&
            ASR::is_a<ASR::StructInstanceMember_t>(*x.m_target) &&
            !ASRUtils::is_character(*asr_value_type)) {
            ASR::StructInstanceMember_t *sim = ASR::down_cast<ASR::StructInstanceMember_t>(x.m_target);
            if (ASRUtils::is_allocatable(sim->m_v)) {
                check_and_allocate(sim->m_v, x.m_value, asr_value_type);
            }
        }

        llvm::Value *target, *value;
        uint32_t h;
        bool lhs_is_string_arrayref = false;
        if( x.m_target->type == ASR::exprType::ArrayItem ||
            x.m_target->type == ASR::exprType::StringItem ||
            x.m_target->type == ASR::exprType::ArraySection ||
            x.m_target->type == ASR::exprType::StructInstanceMember ||
            x.m_target->type == ASR::exprType::ListItem ||
            x.m_target->type == ASR::exprType::DictItem ||
            x.m_target->type == ASR::exprType::UnionInstanceMember ) {
            is_assignment_target = true;
            this->visit_expr(*x.m_target);
            is_assignment_target = false;
            target = tmp;
            if (is_a<ASR::ArrayItem_t>(*x.m_target)) {
                ASR::ArrayItem_t *asr_target0 = ASR::down_cast<ASR::ArrayItem_t>(x.m_target);
                if (is_a<ASR::Var_t>(*asr_target0->m_v)) {
                    ASR::Variable_t *asr_target = ASRUtils::EXPR2VAR(asr_target0->m_v);
                    int n_dims = ASRUtils::extract_n_dims_from_ttype(asr_target->m_type);
                    if ( is_a<ASR::String_t>(*ASRUtils::type_get_past_array(asr_target->m_type)) ) {
                        if (n_dims == 0) {
                            target = llvm_utils->CreateLoad(target);
                            lhs_is_string_arrayref = true;
                        }
                    }
                }
            } else if( ASR::is_a<ASR::StringItem_t>(*x.m_target) ) {
                ASR::StringItem_t *asr_target0 = ASR::down_cast<ASR::StringItem_t>(x.m_target);
                if (is_a<ASR::Var_t>(*asr_target0->m_arg)) {
                    ASR::Variable_t *asr_target = ASRUtils::EXPR2VAR(asr_target0->m_arg);
                    if ( ASRUtils::is_character(*asr_target->m_type) ) {
                        int n_dims = ASRUtils::extract_n_dims_from_ttype(asr_target->m_type);
                        if (n_dims == 0) {
                            lhs_is_string_arrayref = true;
                        }
                    }
                } else if(is_a<ASR::StringPhysicalCast_t>(*asr_target0->m_arg)){ // implies that target is character + n_dim = 0.
                    lhs_is_string_arrayref = true;
                }
            } else if (is_a<ASR::ArraySection_t>(*x.m_target)) {
                ASR::ArraySection_t *asr_target0 = ASR::down_cast<ASR::ArraySection_t>(x.m_target);
                if (is_a<ASR::Var_t>(*asr_target0->m_v)) {
                    ASR::Variable_t *asr_target = ASRUtils::EXPR2VAR(asr_target0->m_v);
                    if ( is_a<ASR::String_t>(*ASRUtils::type_get_past_array(asr_target->m_type)) ) {
                        int n_dims = ASRUtils::extract_n_dims_from_ttype(asr_target->m_type);
                        if (n_dims == 0) {
                            target = llvm_utils->CreateLoad(target);
                            lhs_is_string_arrayref = true;
                        }
                    }
                }
            } else if( ASR::is_a<ASR::ListItem_t>(*x.m_target) ) {
                ASR::ListItem_t* asr_target0 = ASR::down_cast<ASR::ListItem_t>(x.m_target);
                int64_t ptr_loads_copy = ptr_loads;
                ptr_loads = 0;
                this->visit_expr(*asr_target0->m_a);
                ptr_loads = ptr_loads_copy;
                llvm::Value* list = tmp;
                this->visit_expr_wrapper(asr_target0->m_pos, true);
                llvm::Value* pos = tmp;

                target = list_api->read_item_using_ttype(asr_target0->m_type, list, pos, compiler_options.enable_bounds_checking,
                                             module.get(), true);
            }
        } else {
            ASR::Variable_t *asr_target = EXPR2VAR(x.m_target);
            h = get_hash((ASR::asr_t*)asr_target);
            target = llvm_symtab[h];
            if (ASR::is_a<ASR::Pointer_t>(*asr_target->m_type) &&
                !ASR::is_a<ASR::CPtr_t>(
                    *ASRUtils::get_contained_type(asr_target->m_type)) && 
                !ASR::is_a<ASR::StructType_t>(
                    *ASRUtils::get_contained_type(asr_target->m_type)) &&
                !ASRUtils::is_character(*asr_target->m_type)) {
                target = llvm_utils->CreateLoad(target);
            }
            ASR::ttype_t *cont_type = ASRUtils::get_contained_type(asr_target_type);
            if ( ASRUtils::is_array(cont_type) ) {
                if( is_value_list_to_array ) {
                    this->visit_expr_wrapper(x.m_value, true);
                    llvm::Value* list_data = tmp;
                    int64_t ptr_loads_copy = ptr_loads;
                    ptr_loads = 0;
                    this->visit_expr(*ASR::down_cast<ASR::Cast_t>(x.m_value)->m_arg);
                    llvm::Value* plist = tmp;
                    ptr_loads = ptr_loads_copy;
                    llvm::Value* array_data = nullptr;
                    if( ASRUtils::extract_physical_type(asr_target_type) ==
                        ASR::array_physical_typeType::DescriptorArray ) {
                        array_data = llvm_utils->CreateLoad(
                            arr_descr->get_pointer_to_data(llvm_utils->CreateLoad(target)));
                    } else if( ASRUtils::extract_physical_type(asr_target_type) ==
                               ASR::array_physical_typeType::FixedSizeArray ) {
                        array_data = llvm_utils->create_gep(target, 0);
                    } else {
                        LCOMPILERS_ASSERT(false);
                    }
                    llvm::Type* list_llvm_type = llvm_utils->get_type_from_ttype_t_util(x.m_target, asr_target_type, module.get());
                    llvm::Value* size = list_api->len_using_type(list_llvm_type, plist);
                    llvm::Type* el_type = llvm_utils->get_type_from_ttype_t_util(x.m_value,
                        ASRUtils::extract_type(ASRUtils::expr_type(x.m_value)), module.get());
                    llvm::DataLayout data_layout(module->getDataLayout());
                    uint64_t size_ = data_layout.getTypeAllocSize(el_type);
                    size = builder->CreateMul(size, llvm::ConstantInt::get(
                        llvm::Type::getInt32Ty(context), llvm::APInt(32, size_)));
                    builder->CreateMemCpy(array_data, llvm::MaybeAlign(),
                                          list_data, llvm::MaybeAlign(), size);
                    return ;
                }
                if( asr_target->m_type->type == ASR::ttypeType::String ) {
                    target = llvm_utils->CreateLoad(arr_descr->get_pointer_to_data(target));
                }
            }
        }
        if( ASR::is_a<ASR::UnionConstructor_t>(*x.m_value) ) {
            return ;
        }
        ASR::ttype_t* target_type = ASRUtils::expr_type(x.m_target);
        ASR::ttype_t* value_type = ASRUtils::expr_type(x.m_value);
        ASR::expr_t *m_value = x.m_value;
        if (ASRUtils::is_simd_array(x.m_target) && ASR::is_a<ASR::ArraySection_t>(*m_value)) {
            m_value = ASR::down_cast<ASR::ArraySection_t>(m_value)->m_v;
        }
        int ptr_loads_copy = ptr_loads;
        if(ASRUtils::is_character(*value_type)){
            ptr_loads = 0;
        } else if(ASRUtils::is_array(value_type)){
            ptr_loads = 1;
        } else {
            ptr_loads = 2;
        }
        this->visit_expr_wrapper(m_value, true);
        ptr_loads = ptr_loads_copy;
        if( ASR::is_a<ASR::Var_t>(*x.m_value) &&
            ASR::is_a<ASR::UnionType_t>(*value_type) ) {
            tmp = llvm_utils->CreateLoad(tmp);
        }
        value = tmp;
        if (ASR::is_a<ASR::StructType_t>(*target_type) && !ASRUtils::is_class_type(target_type)) {
            if (value->getType()->isPointerTy()) {
                llvm::Type* st_type = llvm_utils->get_type_from_ttype_t_util(x.m_target, target_type, module.get());
                value = llvm_utils->CreateLoad2(st_type, value);
            }
        }
        if ( ASRUtils::is_string_only(ASRUtils::expr_type(x.m_value))) {
            if (lhs_is_string_arrayref) { // Hackish (remove later)
                value = llvm_utils->get_string_data(
                    ASRUtils::get_string_type(asr_value_type), value);
                value = llvm_utils->CreateLoad2(llvm::Type::getInt8Ty(context), value);
                builder->CreateStore(value, target);
            } else {
                llvm_utils->lfortran_str_copy(target, value,
                    ASR::down_cast<ASR::String_t>(ASRUtils::extract_type(asr_target_type)),
                    ASR::down_cast<ASR::String_t>(ASRUtils::extract_type(asr_value_type)),
                    ASRUtils::is_allocatable(asr_target_type));
                tmp = nullptr;
            }
            return;
        }
        if( ASRUtils::is_array(target_type) &&
            ASRUtils::is_array(value_type) &&
            ASRUtils::check_equal_type(target_type, value_type) ) {
            bool data_only_copy = false;
            ASR::array_physical_typeType target_ptype = ASRUtils::extract_physical_type(target_type);
            ASR::array_physical_typeType value_ptype = ASRUtils::extract_physical_type(value_type);
            bool is_target_data_only_array = (target_ptype == ASR::array_physical_typeType::PointerToDataArray);
            bool is_value_data_only_array = (value_ptype == ASR::array_physical_typeType::PointerToDataArray);
            bool is_target_fixed_sized_array = (target_ptype == ASR::array_physical_typeType::FixedSizeArray);
            bool is_value_fixed_sized_array = (value_ptype == ASR::array_physical_typeType::FixedSizeArray);
            bool is_target_simd_array = (target_ptype == ASR::array_physical_typeType::SIMDArray);
            bool is_target_descriptor_based_array = (target_ptype == ASR::array_physical_typeType::DescriptorArray);
            bool is_value_descriptor_based_array = (value_ptype == ASR::array_physical_typeType::DescriptorArray);
            llvm::Type* target_el_type = llvm_utils->get_type_from_ttype_t_util(x.m_target, ASRUtils::extract_type(target_type), module.get());
            llvm::Type* value_el_type = llvm_utils->get_type_from_ttype_t_util(x.m_value, ASRUtils::extract_type(value_type), module.get());
            if( is_value_fixed_sized_array && is_target_fixed_sized_array ) {
                value = llvm_utils->create_gep(value, 0);
                target = llvm_utils->create_gep(target, 0);
                ASR::dimension_t* asr_dims = nullptr;
                size_t asr_n_dims = ASRUtils::extract_dimensions_from_ttype(target_type, asr_dims);
                int64_t size = ASRUtils::get_fixed_size_of_array(asr_dims, asr_n_dims);
                llvm::DataLayout data_layout(module->getDataLayout());
                uint64_t data_size = data_layout.getTypeAllocSize(target_el_type);
                llvm::Value* llvm_size = llvm::ConstantInt::get(context, llvm::APInt(32, size));
                llvm_size = builder->CreateMul(llvm_size,
                    llvm::ConstantInt::get(context, llvm::APInt(32, data_size)));
                builder->CreateMemCpy(target, llvm::MaybeAlign(), value, llvm::MaybeAlign(), llvm_size);
            } else if( is_value_descriptor_based_array && is_target_fixed_sized_array ) {
                value = llvm_utils->CreateLoad2(value_el_type->getPointerTo(), arr_descr->get_pointer_to_data(value));
                target = llvm_utils->create_gep(target, 0);
                ASR::dimension_t* asr_dims = nullptr;
                size_t asr_n_dims = ASRUtils::extract_dimensions_from_ttype(target_type, asr_dims);
                int64_t size = ASRUtils::get_fixed_size_of_array(asr_dims, asr_n_dims);
                llvm::DataLayout data_layout(module->getDataLayout());
                uint64_t data_size = data_layout.getTypeAllocSize(target_el_type);
                llvm::Value* llvm_size = llvm::ConstantInt::get(context, llvm::APInt(32, size));
                llvm_size = builder->CreateMul(llvm_size,
                    llvm::ConstantInt::get(context, llvm::APInt(32, data_size)));
                builder->CreateMemCpy(target, llvm::MaybeAlign(), value, llvm::MaybeAlign(), llvm_size);
            } else if( is_target_descriptor_based_array && is_value_fixed_sized_array ) {
                if( ASRUtils::is_allocatable(target_type) ) {
                    target = llvm_utils->CreateLoad(target);
                }
#if LLVM_VERSION_MAJOR > 16
                ptr_type[target] = llvm_utils->get_type_from_ttype_t_util(x.m_target,
                    ASRUtils::type_get_past_allocatable_pointer(target_type),
                    module.get());
#endif
                llvm::Value* llvm_size = arr_descr->get_array_size(target, nullptr, 4);
                target = llvm_utils->CreateLoad2(target_el_type->getPointerTo(), arr_descr->get_pointer_to_data(target));
                value = llvm_utils->create_gep(value, 0);
                llvm::DataLayout data_layout(module->getDataLayout());
                uint64_t data_size = data_layout.getTypeAllocSize(value_el_type);
                llvm_size = builder->CreateMul(llvm_size,
                    llvm::ConstantInt::get(context, llvm::APInt(32, data_size)));
                builder->CreateMemCpy(target, llvm::MaybeAlign(), value, llvm::MaybeAlign(), llvm_size);
            } else if( is_target_data_only_array || is_value_data_only_array ) {
                if( is_value_fixed_sized_array ) {
                    value = llvm_utils->create_gep(value, 0);
                    is_value_data_only_array = true;
                }
                if( is_target_fixed_sized_array ) {
                    target = llvm_utils->create_gep(target, 0);
                    is_target_data_only_array = true;
                }
                llvm::Value *target_data = nullptr, *value_data = nullptr, *llvm_size = nullptr;
                llvm_size = llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), llvm::APInt(32, 1));
                if( is_target_data_only_array ) {
                    target_data = 
                        ASRUtils::is_array_of_strings(target_type) ?
                        llvm_utils->get_stringArray_data(target_type, target):
                        target;
                    ASR::dimension_t* target_dims = nullptr;
                    int target_ndims = ASRUtils::extract_dimensions_from_ttype(target_type, target_dims);
                    data_only_copy = true;
                    for( int i = 0; i < target_ndims; i++ ) {
                        if( target_dims[i].m_length == nullptr ) {
                            data_only_copy = false;
                            break;
                        }
                        this->visit_expr_wrapper(target_dims[i].m_length, true);
                        llvm_size = builder->CreateMul(llvm_size, builder->CreateSExtOrTrunc(tmp,
                                llvm::Type::getInt32Ty(context)));
                    }
                    if(ASRUtils::is_array_of_strings(target_type)){ // Neglect previous llvm_size
                        llvm_size = llvm_utils->get_stringArray_whole_size(target_type);
                    }
                } else {
                    target_data = 
                            ASRUtils::is_array_of_strings(target_type) ?
                            llvm_utils->get_stringArray_data(target_type, target) :
                            llvm_utils->CreateLoad(arr_descr->get_pointer_to_data(target));
                }
                if( is_value_data_only_array ) {
                    value_data = 
                            ASRUtils::is_array_of_strings(value_type) ?
                            llvm_utils->get_stringArray_data(value_type, value):
                            value;
                    ASR::dimension_t* value_dims = nullptr;
                    int value_ndims = ASRUtils::extract_dimensions_from_ttype(value_type, value_dims);
                    if( !data_only_copy ) {
                        llvm_size = llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), llvm::APInt(32, 1));
                        data_only_copy = true;
                        for( int i = 0; i < value_ndims; i++ ) {
                            if( value_dims[i].m_length == nullptr ) {
                                data_only_copy = false;
                                break;
                            }
                            this->visit_expr_wrapper(value_dims[i].m_length, true);
                            llvm_size = builder->CreateMul(llvm_size, tmp);
                        }
                        if(ASRUtils::is_array_of_strings(value_type)){ // Neglect previous llvm_size
                            llvm_size = llvm_utils->get_stringArray_whole_size(value_type);
                        }
                    }
                } else {
                    llvm::Type* llvm_array_type = llvm_utils->get_type_from_ttype_t_util(x.m_value,
                        ASRUtils::type_get_past_array(
                            ASRUtils::type_get_past_allocatable_pointer(
                                ASRUtils::expr_type(x.m_value))), module.get());
                    value_data = 
                            ASRUtils::is_array_of_strings(value_type) ?
                            llvm_utils->get_stringArray_data(value_type, value) :
                            llvm_utils->CreateLoad2(
                                llvm_array_type->getPointerTo(),
                                arr_descr->get_pointer_to_data(value));
                }
                LCOMPILERS_ASSERT(data_only_copy);
                arr_descr->copy_array_data_only(target_data, value_data, module.get(),
                                                target_el_type,
                                                target_type,
                                                llvm_size);
            } else if ( is_target_simd_array ) {
                if (ASR::is_a<ASR::ArraySection_t>(*x.m_value)) {
                    int idx = 1;
                    ASR::ArraySection_t *arr = down_cast<ASR::ArraySection_t>(x.m_value);
                    (void) ASRUtils::extract_value(arr->m_args->m_left, idx);
                    value = llvm_utils->create_gep(value, idx-1);
                    target = llvm_utils->create_gep(target, 0);
                    ASR::dimension_t* asr_dims = nullptr;
                    size_t asr_n_dims = ASRUtils::extract_dimensions_from_ttype(target_type, asr_dims);
                    int64_t size = ASRUtils::get_fixed_size_of_array(asr_dims, asr_n_dims);
                    llvm::DataLayout data_layout(module->getDataLayout());
                    uint64_t data_size = data_layout.getTypeAllocSize(target_el_type);
                    llvm::Value* llvm_size = llvm::ConstantInt::get(context, llvm::APInt(32, size));
                    llvm_size = builder->CreateMul(llvm_size,
                        llvm::ConstantInt::get(context, llvm::APInt(32, data_size)));
                    builder->CreateMemCpy(target, llvm::MaybeAlign(), value, llvm::MaybeAlign(), llvm_size);
                } else {
                    builder->CreateStore(value, target);
                }
            } else {
                if( LLVM::is_llvm_pointer(*target_type) ) {
                    target = llvm_utils->CreateLoad(target);
                }
#if LLVM_VERSION_MAJOR > 16
                ptr_type[target] = llvm_utils->get_type_from_ttype_t_util(x.m_target,
                    ASRUtils::type_get_past_allocatable_pointer(target_type),
                    module.get());
#endif
                arr_descr->copy_array(value, target, module.get(),
                                      target_type, false);
            }
        } else if( ASR::is_a<ASR::DictItem_t>(*x.m_target) ) {
            ASR::DictItem_t* dict_item_t = ASR::down_cast<ASR::DictItem_t>(x.m_target);
            ASR::Dict_t* dict_type = ASR::down_cast<ASR::Dict_t>(
                                    ASRUtils::expr_type(dict_item_t->m_a));
            int64_t ptr_loads_copy = ptr_loads;
            ptr_loads = 0;
            this->visit_expr(*dict_item_t->m_a);
            llvm::Value* pdict = tmp;

            ptr_loads = !LLVM::is_llvm_struct(dict_type->m_key_type);
            this->visit_expr_wrapper(dict_item_t->m_key, true);
            llvm::Value *key = tmp;
            ptr_loads = ptr_loads_copy;

            llvm_utils->set_dict_api(dict_type);
            // Note - The value is fully loaded to an LLVM value (not at all a pointer)
            // as opposed to DictInsert where LLVM values are loaded depending upon
            // the ASR type of value. Might give issues here.
            llvm_utils->dict_api->write_item(dict_item_t->m_a, pdict, key, value, module.get(),
                                dict_type->m_key_type,
                                dict_type->m_value_type, name2memidx);
        } else if (ASRUtils::is_allocatable(target_type)) {
            if (x.m_realloc_lhs) {
                ASR::ttype_t* asr_type = ASRUtils::type_get_past_pointer(
                    ASRUtils::type_get_past_allocatable(
                    ASRUtils::expr_type(x.m_target)));
                if (ASR::is_a<ASR::Integer_t>(*asr_type) ||
                    ASR::is_a<ASR::Real_t>(*asr_type) ||
                    ASR::is_a<ASR::Complex_t>(*asr_type) ||
                    ASR::is_a<ASR::Logical_t>(*asr_type)) {
                    check_and_allocate(x.m_target);
                }
            }

            target = llvm_utils->CreateLoad(target);
            builder->CreateStore(value, target);
        } else {
            builder->CreateStore(value, target);
        }
    }

    // Checks if target_expr is allocated and if not then allocate
    // Used for compiler_options.po.realloc_lhs
    void check_and_allocate(ASR::expr_t *target_expr, ASR::expr_t *value_expr = nullptr, ASR::ttype_t *value_struct_type = nullptr) {
        LCOMPILERS_ASSERT(compiler_options.po.realloc_lhs);
        ASR::ttype_t *asr_ttype =ASRUtils::expr_type(target_expr);
        ASR::ttype_t *asr_type = ASRUtils::type_get_past_pointer(
            ASRUtils::type_get_past_allocatable(asr_ttype));
        int64_t ptr_loads_copy = ptr_loads;
        ptr_loads = 0;
        this->visit_expr_wrapper(target_expr, false);
        ptr_loads = ptr_loads_copy;
        llvm::Type* llvm_type = llvm_utils->get_type_from_ttype_t_util(target_expr, asr_type, module.get());
        llvm::Value* llvm_cond = nullptr;
        
        if (ASR::is_a<ASR::Allocatable_t>(*asr_ttype) &&
            ASRUtils::is_class_type(ASRUtils::type_get_past_allocatable(asr_ttype))) {
            llvm::Type* target_llvm_type = llvm_utils->get_type_from_ttype_t_util(target_expr, asr_ttype, module.get());
            ASR::ttype_t* wrapped_struct_type = ASRUtils::make_StructType_t_util(al, asr_ttype->base.loc,
                            ASRUtils::symbol_get_past_external(ASRUtils::get_struct_sym_from_struct_expr(target_expr)));
            if (value_struct_type == nullptr) {
                value_struct_type = wrapped_struct_type;
            }
            llvm::Type* wrapper_struct_llvm_type = llvm_utils->get_type_from_ttype_t_util(target_expr, wrapped_struct_type, module.get());
            llvm::Value* target_struct_ptr = llvm_utils->CreateLoad2(wrapper_struct_llvm_type->getPointerTo(),
                                            llvm_utils->create_gep2(target_llvm_type, tmp, 1));
            llvm::Value* null_cond = builder->CreateICmpEQ(
                        target_struct_ptr,
                        llvm::ConstantPointerNull::get(wrapper_struct_llvm_type->getPointerTo()));

            // consider the class hash only if the assignment value is a struct type
            if (ASR::is_a<ASR::StructType_t>(*value_struct_type)) {
                ASR::symbol_t* value_struct_sym = ASRUtils::symbol_get_past_external(ASRUtils::get_struct_sym_from_struct_expr(value_expr));
                llvm::Value* value_hash = llvm::ConstantInt::get(llvm_utils->getIntType(8),
                                llvm::APInt(64, get_class_hash(value_struct_sym)));
                llvm::Value* hash_ptr = llvm_utils->create_gep2(target_llvm_type, tmp, 0);
                llvm::Value* present_hash = llvm_utils->CreateLoad2(llvm_utils->getIntType(8), hash_ptr);
                llvm::Value* hash_cond = builder->CreateICmpNE(
                            present_hash,
                            value_hash);
                // Reallocate when target struct ptr is null or
                // hashes of target and value are not equal
                llvm_cond = builder->CreateOr(null_cond, hash_cond);
            } else {
                // Reallocate when target struct ptr is null
                llvm_cond = null_cond;
            }
        } else {
            llvm_cond = builder->CreateICmpEQ(
                        builder->CreateLoad(llvm_type->getPointerTo(), tmp),
                        llvm::ConstantPointerNull::get(llvm_type->getPointerTo()));
        }
        llvm_utils->create_if_else(
            llvm_cond,
            [&]() {
                    Vec<ASR::alloc_arg_t> alloc_args; alloc_args.reserve(al, 1);
                    ASR::alloc_arg_t alloc_arg;
                    alloc_arg.loc = target_expr->base.loc;
                    alloc_arg.m_a = target_expr;
                    alloc_arg.m_dims = nullptr;
                    alloc_arg.n_dims = 0;
                    alloc_arg.m_sym_subclass = nullptr;
                    // if the assignment target is a class type variable and the value is not a struct type,
                    // then set the ttype of the alloc_arg to the class type
                    if (ASRUtils::is_class_type(asr_type) &&
                        !ASR::is_a<ASR::StructType_t>(*value_struct_type)) {
                        alloc_arg.m_type = ASRUtils::make_StructType_t_util(al, asr_type->base.loc,
                                ASRUtils::symbol_get_past_external(ASRUtils::get_struct_sym_from_struct_expr(target_expr)));
                        alloc_arg.m_sym_subclass = ASRUtils::symbol_get_past_external(ASRUtils::get_struct_sym_from_struct_expr(target_expr));
                    } else {
                        alloc_arg.m_type = value_struct_type;
                        if (value_expr == nullptr) {
                            value_expr = target_expr;
                        }
                        alloc_arg.m_sym_subclass = ASRUtils::symbol_get_past_external(ASRUtils::get_struct_sym_from_struct_expr(value_expr));
                    }
                    alloc_arg.m_len_expr = nullptr;
                    alloc_args.push_back(al, alloc_arg);
                    ASR::stmt_t *alloc_stmt =
                        ASRUtils::STMT(
                            ASR::make_Allocate_t(al, target_expr->base.loc, alloc_args.p, 1, nullptr, nullptr, nullptr));

                    this->visit_stmt(*alloc_stmt);
            },
            []() {});
    }

    void PointerToData_to_Descriptor(ASR::expr_t* expr, ASR::ttype_t* m_type, ASR::ttype_t* m_type_for_dimensions) {
        llvm::Type* target_type = llvm_utils->get_type_from_ttype_t_util(expr,
            ASRUtils::type_get_past_allocatable(
                ASRUtils::type_get_past_pointer(m_type)), module.get());
        llvm::AllocaInst *target = llvm_utils->CreateAlloca(
            target_type, nullptr, "array_descriptor");
        builder->CreateStore(tmp, arr_descr->get_pointer_to_data(target));
        ASR::dimension_t* m_dims = nullptr;
        int n_dims = ASRUtils::extract_dimensions_from_ttype(m_type_for_dimensions, m_dims);
        llvm::Type* llvm_data_type = llvm_utils->get_type_from_ttype_t_util(expr,
            ASRUtils::type_get_past_pointer(ASRUtils::type_get_past_allocatable(m_type)), module.get());
        fill_array_details(target, llvm_data_type, m_dims, n_dims, false, false);
        if( LLVM::is_llvm_pointer(*m_type) ) {
            llvm::AllocaInst* target_ptr = llvm_utils->CreateAlloca(
                target_type->getPointerTo(), nullptr, "array_descriptor_ptr");
            builder->CreateStore(target, target_ptr);
            target = target_ptr;
        }
        tmp = target;
    }

    void visit_ArrayPhysicalCastUtil(llvm::Value* arg, ASR::expr_t* m_arg,
        ASR::ttype_t* m_type, ASR::ttype_t* m_type_for_dimensions,
        ASR::array_physical_typeType m_old, ASR::array_physical_typeType m_new) {

        if( m_old == m_new &&
            m_old != ASR::array_physical_typeType::DescriptorArray ) {
            return ;
        }

        llvm::Type *data_type = llvm_utils->get_type_from_ttype_t_util(m_arg, ASRUtils::extract_type(ASRUtils::expr_type(m_arg)), module.get());
        llvm::Type *arr_type = llvm_utils->get_type_from_ttype_t_util(m_arg,
            ASRUtils::type_get_past_allocatable(
            ASRUtils::type_get_past_pointer(ASRUtils::expr_type(m_arg))),
            module.get());
        llvm::Type* m_arg_llvm_type = llvm_utils->get_type_from_ttype_t_util(m_arg, ASRUtils::expr_type(m_arg), module.get());
        if( m_new == ASR::array_physical_typeType::PointerToDataArray &&
            m_old == ASR::array_physical_typeType::DescriptorArray ) {
            if( ASR::is_a<ASR::StructInstanceMember_t>(*m_arg) ) {
                arg = llvm_utils->CreateLoad2(m_arg_llvm_type, arg);
            }
#if LLVM_VERSION_MAJOR > 16
            ptr_type[arg] = arr_type;
#endif
            tmp = llvm_utils->CreateLoad2(data_type->getPointerTo(), arr_descr->get_pointer_to_data(arg));
            tmp = llvm_utils->create_ptr_gep2(data_type, tmp, arr_descr->get_offset(arg));
        } else if(
            m_new == ASR::array_physical_typeType::PointerToDataArray &&
            m_old == ASR::array_physical_typeType::FixedSizeArray) {
            if( ((ASRUtils::expr_value(m_arg) &&
                !ASR::is_a<ASR::ArrayConstant_t>(*ASRUtils::expr_value(m_arg))) ||
                ASRUtils::expr_value(m_arg) == nullptr ) &&
                !ASR::is_a<ASR::ArrayConstructor_t>(*m_arg) ) {
                tmp = llvm_utils->CreateGEP2(m_arg_llvm_type, tmp, 0);
            }
        } else if(
            m_new == ASR::array_physical_typeType::UnboundedPointerToDataArray &&
            m_old == ASR::array_physical_typeType::FixedSizeArray) {
            if( ((ASRUtils::expr_value(m_arg) &&
                !ASR::is_a<ASR::ArrayConstant_t>(*ASRUtils::expr_value(m_arg))) ||
                ASRUtils::expr_value(m_arg) == nullptr) &&
                !ASR::is_a<ASR::ArrayConstructor_t>(*m_arg) ) {
                tmp = llvm_utils->create_gep2(arr_type, tmp, 0);
            }
        } else if (
            m_new == ASR::array_physical_typeType::SIMDArray &&
            m_old == ASR::array_physical_typeType::FixedSizeArray) {
            // pass
        } else if (
            m_new == ASR::array_physical_typeType::DescriptorArray &&
            m_old == ASR::array_physical_typeType::SIMDArray) {
            tmp = llvm_utils->CreateLoad(arg);
        } else if(
            m_new == ASR::array_physical_typeType::DescriptorArray &&
            m_old == ASR::array_physical_typeType::FixedSizeArray) {
            if( ((ASRUtils::expr_value(m_arg) &&
                !ASR::is_a<ASR::ArrayConstant_t>(*ASRUtils::expr_value(m_arg))) ||
                ASRUtils::expr_value(m_arg) == nullptr) &&
                !ASR::is_a<ASR::ArrayConstructor_t>(*m_arg) ) {
                tmp = llvm_utils->create_gep2(arr_type, tmp, 0);
            }
            PointerToData_to_Descriptor(m_arg, m_type, m_type_for_dimensions);
        } else if(
            m_new == ASR::array_physical_typeType::DescriptorArray &&
            m_old == ASR::array_physical_typeType::PointerToDataArray) {
            PointerToData_to_Descriptor(m_arg, m_type, m_type_for_dimensions);
        } else if(
            m_new == ASR::array_physical_typeType::FixedSizeArray &&
            m_old == ASR::array_physical_typeType::DescriptorArray) {
#if LLVM_VERSION_MAJOR > 16
            ptr_type[arg] = arr_type;
#endif
            tmp = llvm_utils->CreateLoad2(data_type->getPointerTo(), arr_descr->get_pointer_to_data(tmp));
            llvm::Type* target_type = llvm_utils->get_type_from_ttype_t_util(m_arg, m_type, module.get())->getPointerTo();
            tmp = builder->CreateBitCast(tmp, target_type);
        } else if(
            m_new == ASR::array_physical_typeType::DescriptorArray &&
            m_old == ASR::array_physical_typeType::DescriptorArray) {
            // TODO: For allocatables, first check if its allocated (generate code for it)
            // and then if its allocated only then proceed with reseting array details.
            llvm::Type* target_type = llvm_utils->get_type_from_ttype_t_util(m_arg,
                ASRUtils::type_get_past_allocatable(
                    ASRUtils::type_get_past_pointer(m_type)), module.get());
            llvm::AllocaInst *target = llvm_utils->CreateAlloca(
                target_type, nullptr, "array_descriptor");
            builder->CreateStore(llvm_utils->create_ptr_gep2(data_type,
                llvm_utils->CreateLoad2(data_type->getPointerTo(), arr_descr->get_pointer_to_data(tmp)),
                arr_descr->get_offset(tmp)), arr_descr->get_pointer_to_data(target));
            int n_dims = ASRUtils::extract_n_dims_from_ttype(m_type_for_dimensions);
            arr_descr->reset_array_details(target, tmp, n_dims);
            tmp = target;
        } else if (
            m_new == ASR::array_physical_typeType::PointerToDataArray &&
            m_old == ASR::array_physical_typeType::StringArraySinglePointer) {
        //
            if (ASRUtils::is_fixed_size_array(m_type)) {
                if( ((ASRUtils::expr_value(m_arg) &&
                    !ASR::is_a<ASR::ArrayConstant_t>(*ASRUtils::expr_value(m_arg))) ||
                    ASRUtils::expr_value(m_arg) == nullptr) &&
                    !ASR::is_a<ASR::ArrayConstructor_t>(*m_arg) ) {
                    tmp = llvm_utils->create_gep2(arr_type, tmp, 0);
                }
            } else {
                tmp = llvm_utils->CreateLoad(arr_descr->get_pointer_to_data(tmp));
            }
        } else if (
            m_new == ASR::array_physical_typeType::StringArraySinglePointer &&
            m_old == ASR::array_physical_typeType::DescriptorArray) {
            if (ASRUtils::is_fixed_size_array(m_type)) {
                tmp = llvm_utils->CreateLoad(arr_descr->get_pointer_to_data(tmp));
                llvm::Type* target_type = llvm_utils->get_type_from_ttype_t_util(m_arg, m_type, module.get())->getPointerTo();
                tmp = builder->CreateBitCast(tmp, target_type); // [1 x i8*]*
                // we need [1 x i8*]
                tmp = llvm_utils->CreateLoad(tmp);
            }
        } else {
            LCOMPILERS_ASSERT(false);
        }
    }

    void visit_ArrayPhysicalCast(const ASR::ArrayPhysicalCast_t& x) {
        if( x.m_old != ASR::array_physical_typeType::DescriptorArray ) {
            LCOMPILERS_ASSERT(x.m_new != x.m_old);
        }
        int64_t ptr_loads_copy = ptr_loads;
        ptr_loads = 2 - LLVM::is_llvm_pointer(*ASRUtils::expr_type(x.m_arg));
        this->visit_expr_wrapper(x.m_arg, false);
        ptr_loads = ptr_loads_copy;
        visit_ArrayPhysicalCastUtil(tmp, x.m_arg, x.m_type, x.m_type, x.m_old, x.m_new);
    }

    void visit_AssociateBlockCall(const ASR::AssociateBlockCall_t& x) {
        LCOMPILERS_ASSERT(ASR::is_a<ASR::AssociateBlock_t>(*x.m_m));
        ASR::AssociateBlock_t* associate_block = ASR::down_cast<ASR::AssociateBlock_t>(x.m_m);
        declare_vars(*associate_block);
        for (size_t i = 0; i < associate_block->n_body; i++) {
            this->visit_stmt(*(associate_block->m_body[i]));
        }
    }

    void visit_BlockCall(const ASR::BlockCall_t& x) {
        std::vector<llvm::Value*> heap_arrays_copy;
        heap_arrays_copy = heap_arrays;
        heap_arrays.clear();
        if( x.m_label != -1 ) {
            if( llvm_goto_targets.find(x.m_label) == llvm_goto_targets.end() ) {
                llvm::BasicBlock *new_target = llvm::BasicBlock::Create(context, "goto_target");
                llvm_goto_targets[x.m_label] = new_target;
            }
            start_new_block(llvm_goto_targets[x.m_label]);
        }
        LCOMPILERS_ASSERT(ASR::is_a<ASR::Block_t>(*x.m_m));
        ASR::Block_t* block = ASR::down_cast<ASR::Block_t>(x.m_m);
        std::string block_name;
        if (block->m_name) {
            block_name = std::string(block->m_name);
        } else {
            block_name = "block";
        }
        std::string blockstart_name = block_name + ".start";
        std::string blockend_name = block_name + ".end";
        llvm::BasicBlock *blockstart = llvm::BasicBlock::Create(context, blockstart_name);
        start_new_block(blockstart);
        llvm::BasicBlock *blockend = llvm::BasicBlock::Create(context, blockend_name);
        llvm::Function *fn = blockstart->getParent();
#if LLVM_VERSION_MAJOR >= 16
        fn->insert(fn->end(), blockend);
#else
        fn->getBasicBlockList().push_back(blockend);
#endif
        builder->SetInsertPoint(blockstart);
        declare_vars(*block);
        loop_or_block_end.push_back(blockend);
        loop_or_block_end_names.push_back(blockend_name);
        for (size_t i = 0; i < block->n_body; i++) {
            this->visit_stmt(*(block->m_body[i]));
        }
        loop_or_block_end.pop_back();
        loop_or_block_end_names.pop_back();
        llvm::BasicBlock *last_bb = builder->GetInsertBlock();
        llvm::Instruction *block_terminator = last_bb->getTerminator();
        for( auto& value: heap_arrays ) {
            LLVM::lfortran_free(context, *module, *builder, value);
        }
        heap_arrays = heap_arrays_copy;
        if (block_terminator == nullptr) {
            // The previous block is not terminated --- terminate it by jumping
            // to blockend
            builder->CreateBr(blockend);
        }
        builder->SetInsertPoint(blockend);
    }

    inline void visit_expr_wrapper(ASR::expr_t* x, bool load_ref=false, bool is_volatile = false) {
        // Check if *x is nullptr.
        if( x == nullptr ) {
            throw CodeGenError("Internal error: x is nullptr");
        }

        this->visit_expr(*x);
        if( x->type == ASR::exprType::ArrayItem ||
            x->type == ASR::exprType::ArraySection ||
            x->type == ASR::exprType::StructInstanceMember ) {
            if( load_ref &&
                !ASRUtils::is_value_constant(ASRUtils::expr_value(x)) &&
                (ASRUtils::is_array(expr_type(x)) || !ASRUtils::is_character(*expr_type(x)))) {
                llvm::Type* x_llvm_type = llvm_utils->get_type_from_ttype_t_util(x, ASRUtils::expr_type(x), module.get());
                tmp = llvm_utils->CreateLoad2(x_llvm_type, tmp, is_volatile);
            }
        }
    }
    /*
        Visits the expression with the desired pointer loads
        and reverting back the original load.
    */
    void visit_expr_load_wrapper(ASR::expr_t* x, int desired_ptr_load, bool load_ref=false){
        int ptr_loads_copy = ptr_loads;
        ptr_loads = desired_ptr_load;
        if(load_ref){
            this->visit_expr_wrapper(x, true);
        } else {
            this->visit_expr(*x);
        }
        ptr_loads = ptr_loads_copy;
    }

    void fill_type_stmt(const ASR::SelectType_t& x,
        std::vector<ASR::type_stmt_t*>& type_stmt_order,
        ASR::type_stmtType type_stmt_type) {
        for( size_t i = 0; i < x.n_body; i++ ) {
            if( x.m_body[i]->type == type_stmt_type ) {
                type_stmt_order.push_back(x.m_body[i]);
            }
        }
    }

    void visit_SelectType(const ASR::SelectType_t& x) {
        LCOMPILERS_ASSERT(ASR::is_a<ASR::Var_t>(*x.m_selector) || ASR::is_a<ASR::StructInstanceMember_t>(*x.m_selector));
        // Process TypeStmtName first, then ClassStmt
        std::vector<ASR::type_stmt_t*> select_type_stmts;
        fill_type_stmt(x, select_type_stmts, ASR::type_stmtType::TypeStmtName);
        fill_type_stmt(x, select_type_stmts, ASR::type_stmtType::TypeStmtType);
        fill_type_stmt(x, select_type_stmts, ASR::type_stmtType::ClassStmt);
        LCOMPILERS_ASSERT(x.n_body == select_type_stmts.size());
        ASR::Var_t* selector_var = nullptr;
        ASR::StructInstanceMember_t* selector_struct = nullptr;
        std::string selector_var_name;
        if (ASR::is_a<ASR::Var_t>(*x.m_selector)) {
            selector_var = ASR::down_cast<ASR::Var_t>(x.m_selector);
            selector_var_name = ASRUtils::symbol_name(selector_var->m_v);
        } else if (ASR::is_a<ASR::StructInstanceMember_t>(*x.m_selector)) {
            selector_struct = ASR::down_cast<ASR::StructInstanceMember_t>(x.m_selector);
            selector_var_name = ASRUtils::symbol_name(selector_struct->m_m);
        }
        if (x.m_assoc_name) {
            current_selector_var_name = x.m_assoc_name;
        } else {
            current_selector_var_name = selector_var_name;
        }
        uint64_t ptr_loads_copy = ptr_loads;
        ptr_loads = 0;
        if (selector_var) {
            visit_Var(*selector_var);
        } else if (selector_struct) {
            visit_StructInstanceMember(*selector_struct);
        }
        ptr_loads = ptr_loads_copy;
        llvm::Value* llvm_selector = tmp;
        llvm::Type* llvm_selector_type_ = llvm_utils->get_type_from_ttype_t_util(x.m_selector, ASRUtils::expr_type(x.m_selector), module.get());
        llvm::BasicBlock *mergeBB = llvm::BasicBlock::Create(context, "ifcont");
        for( size_t i = 0; i < select_type_stmts.size(); i++ ) {
            llvm::Function *fn = builder->GetInsertBlock()->getParent();

            llvm::BasicBlock *thenBB = llvm::BasicBlock::Create(context, "then", fn);
            llvm::BasicBlock *elseBB = llvm::BasicBlock::Create(context, "else");

            llvm::Value* cond = nullptr;
            ASR::stmt_t** type_block = nullptr;
            size_t n_type_block = 0;
            llvm::Type *i64 = llvm::Type::getInt64Ty(context);
            switch( select_type_stmts[i]->type ) {
                case ASR::type_stmtType::TypeStmtName: {
                    ASR::ttype_t* selector_var_type = ASRUtils::expr_type(x.m_selector);
                    llvm::Value* vptr_int_hash = llvm_utils->CreateLoad2(i64, llvm_utils->create_gep2(llvm_selector_type_, llvm_selector, 0));
                    if( ASRUtils::is_array(selector_var_type) ) {
                        vptr_int_hash = llvm_utils->CreateLoad(llvm_utils->create_gep2(i64, vptr_int_hash, 0));
                    }
                    ASR::TypeStmtName_t* type_stmt_name = ASR::down_cast<ASR::TypeStmtName_t>(select_type_stmts[i]);
                    ASR::symbol_t* type_sym = ASRUtils::symbol_get_past_external(type_stmt_name->m_sym);
                    if( ASR::is_a<ASR::Struct_t>(*type_sym) ) {
                        current_select_type_block_type = llvm_utils->getStructType(
                            ASR::down_cast<ASR::Struct_t>(type_sym), module.get(), false);
                        current_select_type_block_der_type = ASR::down_cast<ASR::Struct_t>(type_sym)->m_name;
                    } else {
                        LCOMPILERS_ASSERT(false);
                    }
                    if ((type2vtab.find(type_sym) == type2vtab.end()) ||
                        (type2vtab[type_sym].find(current_scope) == type2vtab[type_sym].end())) {
                        create_vtab_for_struct_type(type_sym, current_scope);
                    }
                    llvm::Value* type_sym_vtab = type2vtab[type_sym][current_scope];
                    llvm::Type* vtab_struct_type = llvm_utils->getStructType(
                        ASR::down_cast<ASR::Struct_t>(type_sym), module.get(), false);
                    llvm::Value* vtab_obj_casted = builder->CreateBitCast(type_sym_vtab, vtab_struct_type->getPointerTo());
                    cond = builder->CreateICmpEQ(
                            vptr_int_hash,
                            llvm_utils->CreateLoad2( i64, llvm_utils->create_gep2(vtab_struct_type, vtab_obj_casted, 0) ) );
                    type_block = type_stmt_name->m_body;
                    n_type_block = type_stmt_name->n_body;
                    break ;
                }
                case ASR::type_stmtType::ClassStmt: {
                    llvm::Value* vptr_int_hash = llvm_utils->CreateLoad2(i64, llvm_utils->create_gep2(llvm_selector_type_, llvm_selector, 0));
                    ASR::ClassStmt_t* class_stmt = ASR::down_cast<ASR::ClassStmt_t>(select_type_stmts[i]);
                    ASR::symbol_t* class_sym = ASRUtils::symbol_get_past_external(class_stmt->m_sym);
                    if( ASR::is_a<ASR::Struct_t>(*class_sym) ) {
                        current_select_type_block_type = llvm_utils->getStructType(
                            ASR::down_cast<ASR::Struct_t>(class_sym), module.get(), false);
                        current_select_type_block_der_type = ASR::down_cast<ASR::Struct_t>(class_sym)->m_name;
                    } else {
                        LCOMPILERS_ASSERT(false);
                    }

                    std::vector<llvm::Value*>& class_sym_vtabs = class2vtab[class_sym][current_scope];
                    std::vector<llvm::Value*> conds;
                    if (class_sym_vtabs.size() == 0) {
                        if ((type2vtab.find(class_sym) == type2vtab.end()) ||
                            (type2vtab[class_sym].find(current_scope) == type2vtab[class_sym].end())) {
                            create_vtab_for_struct_type(class_sym, current_scope);
                        }
                        class_sym_vtabs.push_back(type2vtab[class_sym][current_scope]);
                    }
                    conds.reserve(class_sym_vtabs.size());
                    for( size_t i = 0; i < class_sym_vtabs.size(); i++ ) {
                        llvm::Value* vtab_obj_casted = builder->CreateBitCast(
                            class_sym_vtabs[i], current_select_type_block_type->getPointerTo());
                        conds.push_back(builder->CreateICmpEQ(
                            vptr_int_hash,
                            llvm_utils->CreateLoad2(i64, llvm_utils->create_gep2(current_select_type_block_type, vtab_obj_casted, 0)) ));
                    }
                    cond = builder->CreateOr(conds);
                    type_block = class_stmt->m_body;
                    n_type_block = class_stmt->n_body;
                    break ;
                }
                case ASR::type_stmtType::TypeStmtType: {
                    ASR::ttype_t* selector_var_type = ASRUtils::expr_type(x.m_selector);
                    ASR::TypeStmtType_t* type_stmt_type_t = ASR::down_cast<ASR::TypeStmtType_t>(select_type_stmts[i]);
                    ASR::ttype_t* type_stmt_type = type_stmt_type_t->m_type;
                    current_select_type_block_type = llvm_utils->get_type_from_ttype_t_util(x.m_selector, type_stmt_type, module.get());
                    current_select_type_block_type_asr = type_stmt_type;
                    llvm::Value* intrinsic_type_id = llvm::ConstantInt::get(llvm_utils->getIntType(8),
                        llvm::APInt(64, -((int) type_stmt_type->type) -
                            ASRUtils::extract_kind_from_ttype_t(type_stmt_type), true));
                    llvm::Value* _type_id = nullptr;
                    if( ASRUtils::is_array(selector_var_type) ) {
                        llvm::Type* el_type = llvm_utils->get_type_from_ttype_t_util(x.m_selector, ASRUtils::extract_type(selector_var_type), module.get());
                        llvm::Value* data_ptr = llvm_utils->CreateLoad2(el_type->getPointerTo(), arr_descr->get_pointer_to_data(llvm_selector));
                        _type_id = llvm_utils->CreateLoad2(llvm::Type::getInt64Ty(context), llvm_utils->create_gep2(el_type, data_ptr, 0));
                    } else {
                        _type_id = llvm_utils->CreateLoad2(llvm::Type::getInt64Ty(context), llvm_utils->create_gep2(llvm_selector_type_, llvm_selector, 0));
                    }
                    cond = builder->CreateICmpEQ(_type_id, intrinsic_type_id);
                    type_block = type_stmt_type_t->m_body;
                    n_type_block = type_stmt_type_t->n_body;
                    break;
                }
                default: {
                    throw CodeGenError("ASR::type_stmtType, " +
                                       std::to_string(x.m_body[i]->type) +
                                       " is not yet supported.");
                }
            }
            builder->CreateCondBr(cond, thenBB, elseBB);
            builder->SetInsertPoint(thenBB);
            // TODO: change symtab for arrays too
            bool change_symtab = ASRUtils::is_unlimited_polymorphic_type(x.m_selector)
                && !ASRUtils::is_array(ASRUtils::expr_type(x.m_selector));
            // For class(*) selector variables cast to current select block type and update llvm_symtab temporarily
            // while executing blocks.
            if (change_symtab) {
                llvm::Type* src_type = llvm_utils->get_type_from_ttype_t_util(x.m_selector, expr_type(x.m_selector), module.get());
                llvm::Value* llvm_selector_new = llvm_utils->create_gep2(src_type, llvm_selector, 1);
                llvm_selector_new = llvm_utils->CreateLoad2(llvm::Type::getVoidTy(context)->getPointerTo(),llvm_selector_new);
                llvm_selector_new = builder->CreateBitCast(llvm_selector_new, current_select_type_block_type->getPointerTo());
                if (current_select_type_block_type_asr &&
                    !ASRUtils::is_character(*current_select_type_block_type_asr)) {
                    llvm_selector_new = llvm_utils->CreateLoad2(current_select_type_block_type, llvm_selector_new);
                }
                uint32_t h = get_hash((ASR::asr_t*)ASR::down_cast<ASR::Variable_t>(current_scope->resolve_symbol(selector_var_name)));
                llvm_symtab[h] = llvm_selector_new;
            }
            {
                if( n_type_block == 1 && ASR::is_a<ASR::BlockCall_t>(*type_block[0]) ) {
                    ASR::BlockCall_t* block_call = ASR::down_cast<ASR::BlockCall_t>(type_block[0]);
                    ASR::Block_t* block_t = ASR::down_cast<ASR::Block_t>(block_call->m_m);
                    declare_vars(*block_t, false);
                    for( size_t j = 0; j < block_t->n_body; j++ ) {
                        this->visit_stmt(*block_t->m_body[j]);
                    }
                }
            }
            builder->CreateBr(mergeBB);
            if (change_symtab) {
                uint32_t h = get_hash((ASR::asr_t*)ASR::down_cast<ASR::Variable_t>(current_scope->resolve_symbol(selector_var_name)));
                llvm_symtab[h] = llvm_selector;
            }

            start_new_block(elseBB);
            current_select_type_block_type = nullptr;
            current_select_type_block_type_asr = nullptr;
            current_select_type_block_der_type.clear();
        }
        if( x.n_default > 0 ) {
            for( size_t i = 0; i < x.n_default; i++ ) {
                this->visit_stmt(*x.m_default[i]);
            }
        }
        start_new_block(mergeBB);
        current_selector_var_name.clear();
    }

    void visit_IntegerCompare(const ASR::IntegerCompare_t &x) {
        if (x.m_value) {
            this->visit_expr_wrapper(x.m_value, true);
            return;
        }
        this->visit_expr_load_wrapper(x.m_left,
            LLVM::is_llvm_pointer(*expr_type(x.m_left)) ? 2 : 1,
            true);
        llvm::Value *left = tmp;
        this->visit_expr_load_wrapper(x.m_right,
            LLVM::is_llvm_pointer(*expr_type(x.m_right)) ? 2 : 1,
            true);
        llvm::Value *right = tmp;
        load_non_array_non_character_pointers(x.m_left, ASRUtils::expr_type(x.m_left), left);
        load_non_array_non_character_pointers(x.m_right, ASRUtils::expr_type(x.m_right), right);
        switch (x.m_op) {
            case (ASR::cmpopType::Eq) : {
                tmp = builder->CreateICmpEQ(left, right);
                break;
            }
            case (ASR::cmpopType::Gt) : {
                tmp = builder->CreateICmpSGT(left, right);
                break;
            }
            case (ASR::cmpopType::GtE) : {
                tmp = builder->CreateICmpSGE(left, right);
                break;
            }
            case (ASR::cmpopType::Lt) : {
                tmp = builder->CreateICmpSLT(left, right);
                break;
            }
            case (ASR::cmpopType::LtE) : {
                tmp = builder->CreateICmpSLE(left, right);
                break;
            }
            case (ASR::cmpopType::NotEq) : {
                tmp = builder->CreateICmpNE(left, right);
                break;
            }
            default : {
                throw CodeGenError("Comparison operator not implemented",
                        x.base.base.loc);
            }
        }
    }

    void visit_UnsignedIntegerCompare(const ASR::UnsignedIntegerCompare_t &x) {
        if (x.m_value) {
            this->visit_expr_wrapper(x.m_value, true);
            return;
        }
        this->visit_expr_wrapper(x.m_left, true);
        llvm::Value *left = tmp;
        this->visit_expr_wrapper(x.m_right, true);
        llvm::Value *right = tmp;
        switch (x.m_op) {
            case (ASR::cmpopType::Eq) : {
                tmp = builder->CreateICmpEQ(left, right);
                break;
            }
            case (ASR::cmpopType::Gt) : {
                tmp = builder->CreateICmpUGT(left, right);
                break;
            }
            case (ASR::cmpopType::GtE) : {
                tmp = builder->CreateICmpUGE(left, right);
                break;
            }
            case (ASR::cmpopType::Lt) : {
                tmp = builder->CreateICmpULT(left, right);
                break;
            }
            case (ASR::cmpopType::LtE) : {
                tmp = builder->CreateICmpULE(left, right);
                break;
            }
            case (ASR::cmpopType::NotEq) : {
                tmp = builder->CreateICmpNE(left, right);
                break;
            }
            default : {
                throw CodeGenError("Comparison operator not implemented",
                        x.base.base.loc);
            }
        }
    }

    void visit_CPtrCompare(const ASR::CPtrCompare_t &x) {
        if (x.m_value) {
            this->visit_expr_wrapper(x.m_value, true);
            return;
        }
        this->visit_expr_wrapper(x.m_left, true);
        llvm::Value *left = tmp;
        left = builder->CreatePtrToInt(left, llvm_utils->getIntType(8, false));
        this->visit_expr_wrapper(x.m_right, true);
        llvm::Value *right = tmp;
        right = builder->CreatePtrToInt(right, llvm_utils->getIntType(8, false));
        switch (x.m_op) {
            case (ASR::cmpopType::Eq) : {
                tmp = builder->CreateICmpEQ(left, right);
                break;
            }
            case (ASR::cmpopType::Gt) : {
                tmp = builder->CreateICmpSGT(left, right);
                break;
            }
            case (ASR::cmpopType::GtE) : {
                tmp = builder->CreateICmpSGE(left, right);
                break;
            }
            case (ASR::cmpopType::Lt) : {
                tmp = builder->CreateICmpSLT(left, right);
                break;
            }
            case (ASR::cmpopType::LtE) : {
                tmp = builder->CreateICmpSLE(left, right);
                break;
            }
            case (ASR::cmpopType::NotEq) : {
                tmp = builder->CreateICmpNE(left, right);
                break;
            }
            default : {
                throw CodeGenError("Comparison operator not implemented",
                        x.base.base.loc);
            }
        }
    }

    void visit_RealCompare(const ASR::RealCompare_t &x) {
        if (x.m_value) {
            this->visit_expr_wrapper(x.m_value, true);
            return;
        }
        this->visit_expr_wrapper(x.m_left, true);
        llvm::Value *left = tmp;
        this->visit_expr_wrapper(x.m_right, true);
        llvm::Value *right = tmp;
        switch (x.m_op) {
            case (ASR::cmpopType::Eq) : {
                tmp = builder->CreateFCmpOEQ(left, right);
                break;
            }
            case (ASR::cmpopType::Gt) : {
                tmp = builder->CreateFCmpOGT(left, right);
                break;
            }
            case (ASR::cmpopType::GtE) : {
                tmp = builder->CreateFCmpOGE(left, right);
                break;
            }
            case (ASR::cmpopType::Lt) : {
                tmp = builder->CreateFCmpOLT(left, right);
                break;
            }
            case (ASR::cmpopType::LtE) : {
                tmp = builder->CreateFCmpOLE(left, right);
                break;
            }
            case (ASR::cmpopType::NotEq) : {
                tmp = builder->CreateFCmpUNE(left, right);
                break;
            }
            default : {
                throw CodeGenError("Comparison operator not implemented",
                        x.base.base.loc);
            }
        }
    }

    void visit_ComplexCompare(const ASR::ComplexCompare_t &x) {
        if (x.m_value) {
            this->visit_expr_wrapper(x.m_value, true);
            return;
        }
        this->visit_expr_wrapper(x.m_left, true);
        llvm::Value *left = tmp;
        this->visit_expr_wrapper(x.m_right, true);
        llvm::Value *right = tmp;
        llvm::Value* real_left = complex_re(left, left->getType());
        llvm::Value* real_right = complex_re(right, right->getType());
        llvm::Value* img_left = complex_im(left, left->getType());
        llvm::Value* img_right = complex_im(right, right->getType());
        llvm::Value *real_res, *img_res;
        switch (x.m_op) {
            case (ASR::cmpopType::Eq) : {
                real_res = builder->CreateFCmpOEQ(real_left, real_right);
                img_res = builder->CreateFCmpOEQ(img_left, img_right);
                tmp = builder->CreateAnd(real_res, img_res);
                break;
            }
            case (ASR::cmpopType::NotEq) : {
                real_res = builder->CreateFCmpONE(real_left, real_right);
                img_res = builder->CreateFCmpONE(img_left, img_right);
                tmp = builder->CreateOr(real_res, img_res);
                break;
            }
            default : {
                throw CodeGenError("Comparison operator not implemented",
                        x.base.base.loc);
            }
        }
    }

    void visit_StringCompare(const ASR::StringCompare_t &x) {
        if (x.m_value) {
            this->visit_expr_wrapper(x.m_value, true);
            return;
        }
        llvm::Value *left, *left_len;
        llvm::Value *right , *right_len;
        std::tie(left, left_len) = get_string_data_and_length(x.m_left);
        std::tie(right, right_len) = get_string_data_and_length(x.m_right);

        bool is_single_char;
        { // Set `is_single_char` flag
            ASR::String_t* left_str = ASRUtils::get_string_type(x.m_left);
            ASR::String_t* right_str = ASRUtils::get_string_type(x.m_right);
            int64_t l_str_len, r_str_len;
            if( ASRUtils::is_value_constant(left_str->m_len, l_str_len) &&
                ASRUtils::is_value_constant(right_str->m_len, r_str_len)){// Check + Fetch len
                is_single_char = (l_str_len == 1) && (r_str_len == 1);
            } else {
                is_single_char = false;
            }
        }
        if( is_single_char ) {
            left = llvm_utils->CreateLoad2(llvm::Type::getInt8Ty(context), left);
            right = llvm_utils->CreateLoad2(llvm::Type::getInt8Ty(context), right);
        } else {
            tmp = lfortran_str_cmp(left, left_len, right, right_len);
        }
        switch (x.m_op) {
            case (ASR::cmpopType::Eq) : {
                if( is_single_char ) {
                    tmp = builder->CreateICmpEQ(left, right);
                } else {
                    tmp = builder->CreateICmpEQ(tmp, llvm::ConstantInt::get(context, llvm::APInt(32,0)));
                }
                break;
            }
            case (ASR::cmpopType::NotEq) : {
                if( is_single_char ) {
                    tmp = builder->CreateICmpNE(left, right);
                } else {
                    tmp = builder->CreateICmpNE(tmp, llvm::ConstantInt::get(context, llvm::APInt(32,0)));
                }
                break;
            }
            case (ASR::cmpopType::Gt) : {
                if( is_single_char ) {
                    tmp = builder->CreateICmpSGT(left, right);
                } else {
                    tmp = builder->CreateICmpSGT(tmp, llvm::ConstantInt::get(context, llvm::APInt(32,0)));
                }
                break;
            }
            case (ASR::cmpopType::GtE) : {
                if( is_single_char ) {
                    tmp = builder->CreateICmpSGE(left, right);
                } else {
                    tmp = builder->CreateICmpSGE(tmp, llvm::ConstantInt::get(context, llvm::APInt(32,0)));
                }
                break;
            }
            case (ASR::cmpopType::Lt) : {
                if( is_single_char ) {
                    tmp = builder->CreateICmpSLT(left, right);
                } else {
                    tmp = builder->CreateICmpSLT(tmp, llvm::ConstantInt::get(context, llvm::APInt(32,0)));
                }
                break;
            }
            case (ASR::cmpopType::LtE) : {
                if( is_single_char ) {
                    tmp = builder->CreateICmpSLE(left, right);
                } else {
                    tmp = builder->CreateICmpSLE(tmp, llvm::ConstantInt::get(context, llvm::APInt(32,0)));
                }
                break;
            }
            default : {
                throw CodeGenError("Comparison operator not implemented",
                        x.base.base.loc);
            }
        }
    }

    void visit_LogicalCompare(const ASR::LogicalCompare_t &x) {
        if (x.m_value) {
            this->visit_expr_wrapper(x.m_value, true);
            return;
        }
        this->visit_expr_wrapper(x.m_left, true);
        llvm::Value *left = tmp;
        this->visit_expr_wrapper(x.m_right, true);
        llvm::Value *right = tmp;
        // i1 -> i32
        left = builder->CreateZExt(left, llvm::Type::getInt32Ty(context));
        right = builder->CreateZExt(right, llvm::Type::getInt32Ty(context));
        switch (x.m_op) {
            case (ASR::cmpopType::Eq) : {
                tmp = builder->CreateICmpEQ(left, right);
                break;
            }
            case (ASR::cmpopType::NotEq) : {
                tmp = builder->CreateICmpNE(left, right);
                break;
            }
            case (ASR::cmpopType::Gt) : {
                tmp = builder->CreateICmpUGT(left, right);
                break;
            }
            case (ASR::cmpopType::GtE) : {
                tmp = builder->CreateICmpUGE(left, right);
                break;
            }
            case (ASR::cmpopType::Lt) : {
                tmp = builder->CreateICmpULT(left, right);
                break;
            }
            case (ASR::cmpopType::LtE) : {
                tmp = builder->CreateICmpULE(left, right);
                break;
            }
            default : {
                throw CodeGenError("Comparison operator not implemented",
                        x.base.base.loc);
            }
        }
    }

    void visit_OverloadedCompare(const ASR::OverloadedCompare_t &x) {
        this->visit_expr(*x.m_overloaded);
    }

    void visit_If(const ASR::If_t &x) {
        llvm::Value **strings_to_be_deallocated_copy = strings_to_be_deallocated.p;
        size_t n = strings_to_be_deallocated.n;
        strings_to_be_deallocated.reserve(al, 1);
        this->visit_expr_wrapper(x.m_test, true);
        llvm_utils->create_if_else(tmp, [&]() {
            for (size_t i=0; i<x.n_body; i++) {
                this->visit_stmt(*x.m_body[i]);
            }
        }, [&]() {
            for (size_t i=0; i<x.n_orelse; i++) {
                this->visit_stmt(*x.m_orelse[i]);
            }
        }, x.m_name, loop_or_block_end, loop_or_block_end_names);
        strings_to_be_deallocated.reserve(al, n);
        strings_to_be_deallocated.n = n;
        strings_to_be_deallocated.p = strings_to_be_deallocated_copy;
    }

    void visit_IfExp(const ASR::IfExp_t &x) {
        // IfExp(expr test, expr body, expr orelse, ttype type, expr? value)
        this->visit_expr_wrapper(x.m_test, true);
        llvm::Value *cond = tmp;
        llvm::Type* _type = llvm_utils->get_type_from_ttype_t_util(x.m_test, x.m_type, module.get());
        llvm::Value* ifexp_res = llvm_utils->CreateAlloca(_type, nullptr, "");
        llvm_utils->create_if_else(cond, [&]() {
            this->visit_expr_wrapper(x.m_body, true);
            builder->CreateStore(tmp, ifexp_res);
        }, [&]() {
            this->visit_expr_wrapper(x.m_orelse, true);
            builder->CreateStore(tmp, ifexp_res);
        });
        tmp = llvm_utils->CreateLoad(ifexp_res);
    }

    // TODO: Implement visit_DooLoop
    //void visit_DoLoop(const ASR::DoLoop_t &x) {
    //}

    void visit_WhileLoop(const ASR::WhileLoop_t &x) {
        llvm::Value **strings_to_be_deallocated_copy = strings_to_be_deallocated.p;
        size_t n = strings_to_be_deallocated.n;
        strings_to_be_deallocated.reserve(al, 1);
        create_loop(x.m_name, [=]() {
            this->visit_expr_wrapper(x.m_test, true);
            return tmp;
        }, [&]() {
            for (size_t i=0; i<x.n_body; i++) {
                this->visit_stmt(*x.m_body[i]);
            }
        });
        strings_to_be_deallocated.reserve(al, n);
        strings_to_be_deallocated.n = n;
        strings_to_be_deallocated.p = strings_to_be_deallocated_copy;
    }

    bool case_insensitive_string_compare(const std::string& str1, const std::string& str2) {
        if (str1.size() != str2.size()) {
            return false;
        }
        for (std::string::const_iterator c1 = str1.begin(), c2 = str2.begin(); c1 != str1.end(); ++c1, ++c2) {
            if (tolower(static_cast<unsigned char>(*c1)) != tolower(static_cast<unsigned char>(*c2))) {
                return false;
            }
        }
        return true;
    }

    void visit_Exit(const ASR::Exit_t &x) {
        if (x.m_stmt_name) {
            std::string stmt_name = std::string(x.m_stmt_name) + ".end";
            std::string if_cont_name = std::string(x.m_stmt_name) + ".ifcont";
            int nested_block_depth = loop_or_block_end_names.size();
            int i = nested_block_depth - 1;
            for (; i >= 0; i--) {
                if (case_insensitive_string_compare(loop_or_block_end_names[i], stmt_name) ||
                    case_insensitive_string_compare(loop_or_block_end_names[i], if_cont_name)) {
                    break;
                }
            }
            if (i >= 0) {
                builder->CreateBr(loop_or_block_end[i]);
            } else {
                throw CodeGenError("Could not find block or loop named " + std::string(x.m_stmt_name) + " in parent scope to exit from.",
                x.base.base.loc);
            }
        } else {
            builder->CreateBr(loop_or_block_end.back());
        }
        llvm::BasicBlock *bb = llvm::BasicBlock::Create(context, "unreachable_after_exit");
        start_new_block(bb);
    }

    void visit_Cycle(const ASR::Cycle_t &x) {
        if (x.m_stmt_name) {
            std::string stmt_name = std::string(x.m_stmt_name) + ".head";
            int nested_block_depth = loop_head_names.size();
            int i = nested_block_depth - 1;
            for (; i >= 0; i--) {
                if (case_insensitive_string_compare(loop_head_names[i], stmt_name)) {
                    break;
                }
            }
            if (i >= 0) {
                builder->CreateBr(loop_head[i]);
            } else {
                throw CodeGenError("Could not find loop named " + std::string(x.m_stmt_name) + " in parent scope to cycle to.",
                x.base.base.loc);
            }
        } else {
            builder->CreateBr(loop_head.back());
        }
        llvm::BasicBlock *bb = llvm::BasicBlock::Create(context, "unreachable_after_cycle");
        start_new_block(bb);
    }

    void visit_Return(const ASR::Return_t & /* x */) {
        builder->CreateBr(proc_return);
        llvm::BasicBlock *bb = llvm::BasicBlock::Create(context, "unreachable_after_return");
        start_new_block(bb);
    }

    void visit_GoTo(const ASR::GoTo_t &x) {
        if (llvm_goto_targets.find(x.m_target_id) == llvm_goto_targets.end()) {
            // If the target does not exist yet, create it
            llvm::BasicBlock *new_target = llvm::BasicBlock::Create(context, "goto_target");
            llvm_goto_targets[x.m_target_id] = new_target;
        }
        llvm::BasicBlock *target = llvm_goto_targets[x.m_target_id];
        builder->CreateBr(target);
        llvm::BasicBlock *bb = llvm::BasicBlock::Create(context, "unreachable_after_goto");
        start_new_block(bb);
    }

    void visit_GoToTarget(const ASR::GoToTarget_t &x) {
        if (llvm_goto_targets.find(x.m_id) == llvm_goto_targets.end()) {
            // If the target does not exist yet, create it
            llvm::BasicBlock *new_target = llvm::BasicBlock::Create(context, "goto_target");
            llvm_goto_targets[x.m_id] = new_target;
        }
        llvm::BasicBlock *target = llvm_goto_targets[x.m_id];
        start_new_block(target);
    }

    void visit_LogicalBinOp(const ASR::LogicalBinOp_t &x) {
        if (x.m_value) {
            this->visit_expr_wrapper(x.m_value, true);
            return;
        }
        this->visit_expr_wrapper(x.m_left, true);
        llvm::Value *left_val = tmp;
        this->visit_expr_wrapper(x.m_right, true);
        llvm::Value *right_val = tmp;
        llvm::Value *zero, *cond;
        if (ASRUtils::is_integer(*x.m_type)) {
            int a_kind = down_cast<ASR::Integer_t>(x.m_type)->m_kind;
            int init_value_bits = 8*a_kind;
            zero = llvm::ConstantInt::get(context,
                            llvm::APInt(init_value_bits, 0));
            cond = builder->CreateICmpEQ(left_val, zero);
        } else if (ASRUtils::is_real(*x.m_type)) {
            int a_kind = down_cast<ASR::Real_t>(x.m_type)->m_kind;
            int init_value_bits = 8*a_kind;
            if (init_value_bits == 32) {
                zero = llvm::ConstantFP::get(context,
                                    llvm::APFloat((float)0));
            } else {
                zero = llvm::ConstantFP::get(context,
                                    llvm::APFloat((double)0));
            }
            cond = builder->CreateFCmpUEQ(left_val, zero);
        } else if (ASRUtils::is_character(*x.m_type)) {
            zero = llvm::Constant::getNullValue(character_type);
            llvm::Value* comp_res =lfortran_str_cmp(left_val, get_string_length(x.m_left),
                zero, llvm::ConstantInt::get(context, llvm::APInt(64, 0)));
            cond = builder->CreateICmpEQ(
                comp_res, 
                llvm::ConstantInt::get(context, llvm::APInt(32, 0)));
        } else if (ASRUtils::is_logical(*x.m_type)) {
            zero = llvm::ConstantInt::get(context,
                            llvm::APInt(1, 0));
            cond = builder->CreateICmpEQ(left_val, zero);
        } else {
            throw CodeGenError("Only Integer, Real, Strings and Logical types are supported "
            "in logical binary operation.", x.base.base.loc);
        }
        switch (x.m_op) {
            case ASR::logicalbinopType::And: {
                tmp = builder->CreateSelect(cond, left_val, right_val);
                break;
            };
            case ASR::logicalbinopType::Or: {
                tmp = builder->CreateSelect(cond, right_val, left_val);
                break;
            };
            case ASR::logicalbinopType::Xor: {
                tmp = builder->CreateXor(left_val, right_val);
                break;
            };
            case ASR::logicalbinopType::NEqv: {
                tmp = builder->CreateXor(left_val, right_val);
                break;
            };
            case ASR::logicalbinopType::Eqv: {
                tmp = builder->CreateXor(left_val, right_val);
                tmp = builder->CreateNot(tmp);
            };
        }
    }

    void visit_StringRepeat(const ASR::StringRepeat_t &x) {
        if (x.m_value) {
            this->visit_expr_wrapper(x.m_value, true);
            return;
        }
        this->visit_expr_wrapper(x.m_left, true);
        llvm::Value *left_val = tmp;
        this->visit_expr_wrapper(x.m_right, true);
        llvm::Value *right_val = tmp;
        tmp = lfortran_strrepeat(left_val, right_val);
    }

    void visit_StringConcat(const ASR::StringConcat_t &x) {
        if (x.m_value) {
            this->visit_expr_wrapper(x.m_value, true);
            return;
        }
        llvm::Value* left_val {}, *left_len {};
        llvm::Value* right_val {}, *right_len {};
        std::tie(left_val, left_len) = get_string_data_and_length(x.m_left);
        std::tie(right_val, right_len) = get_string_data_and_length(x.m_right);
        tmp = lfortran_strConcat(left_val, left_len, right_val, right_len);
        tmp = llvm_utils->create_string_descriptor(tmp,
            builder->CreateAdd(left_len, right_len), "strConcat_desc");
    }

    void visit_StringLen(const ASR::StringLen_t &x) {
        if (x.m_value) {
            this->visit_expr_wrapper(x.m_value, true);
            return;
        }
        tmp = get_string_length(x.m_arg); // `int64` value.
        tmp = llvm_utils->convert_kind(tmp,
            llvm::Type::getIntNTy(context, ASRUtils::extract_kind_from_ttype_t(x.m_type) * 8));
    }

    void visit_StringOrd(const ASR::StringOrd_t &x) {
        if (x.m_value) {
            this->visit_expr_wrapper(x.m_value, true);
            return;
        }
        this->visit_expr_wrapper(x.m_arg, true);
        llvm::AllocaInst *parg = llvm_utils->CreateAlloca(*builder, character_type);
        builder->CreateStore(tmp, parg);
        tmp = lfortran_str_ord(parg);
    }

    void visit_StringChr(const ASR::StringChr_t &x) {
        if (x.m_value) {
            this->visit_expr_wrapper(x.m_value, true);
            return;
        }
        this->visit_expr_load_wrapper(x.m_arg, 1);
        tmp = lfortran_str_chr(tmp);
        tmp = llvm_utils->create_string_descriptor(tmp,
            llvm::ConstantInt::get(context, llvm::APInt(64, 1)), "stringChr_desc");
    }

    void visit_StringItem(const ASR::StringItem_t& x) {
        if (x.m_value) {
            this->visit_expr_wrapper(x.m_value, true);
            return;
        }
        llvm::Value *idx {};
        llvm::Value *str {};
        { // Set proper load + Visit + Revert load
            int64_t ptr_loads_copy = ptr_loads;
            ptr_loads = 1; this->visit_expr_wrapper(x.m_idx, true);
            idx = tmp;
            ptr_loads = 0; this->visit_expr_wrapper(x.m_arg, true);
            str = tmp;
            ptr_loads = ptr_loads_copy;
        }
        if( is_assignment_target ) { // Hackish
            idx = builder->CreateSub(builder->CreateSExtOrTrunc(idx, llvm::Type::getInt32Ty(context)),
                llvm::ConstantInt::get(context, llvm::APInt(32, 1)));
            std::vector<llvm::Value*> idx_vec = {idx};
            llvm::Value* str_data = llvm_utils->get_string_data(ASRUtils::get_string_type(x.m_arg), str);
            tmp = llvm_utils->CreateGEP2(llvm::Type::getInt8Ty(context), str_data, idx_vec);
        } else {
            // TODO : Create a string (depending on the physicalType) of length 1 using LLVM, instead of calling runtime C function
            llvm::Value* str_data{}, *str_len{};
            std::tie(str_data, str_len) = llvm_utils->get_string_length_data(ASRUtils::get_string_type(x.m_arg), str);
            tmp = lfortran_str_item(str_data, str_len, idx);
            tmp = llvm_utils->create_string_descriptor(tmp, 
                llvm::ConstantInt::get(context, llvm::APInt(64, 1)), "stringItem_desc");
        }
    }

    void stringSection_helper_python(const ASR::StringSection_t &x){
        this->visit_expr_load_wrapper(x.m_arg, 0);
        llvm::Value *str = tmp;
        llvm::Value *left, *right, *step;
        llvm::Value *left_present, *right_present;
        if (x.m_start) {
            this->visit_expr_load_wrapper(x.m_start, 1);
            left = tmp;
            left_present = llvm::ConstantInt::get(context,
                llvm::APInt(1, 1));
        } else {
            left = llvm::Constant::getNullValue(llvm::Type::getInt32Ty(context));
            left_present = llvm::ConstantInt::get(context,
                llvm::APInt(1, 0));
        }
        if (x.m_end) {
            this->visit_expr_load_wrapper(x.m_end, 1);
            right = tmp;
            right_present = llvm::ConstantInt::get(context,
                llvm::APInt(1, 1));
        } else {
            right = llvm::Constant::getNullValue(llvm::Type::getInt32Ty(context));
            right_present = llvm::ConstantInt::get(context,
                llvm::APInt(1, 0));
        }
        if (x.m_step) {
            this->visit_expr_load_wrapper(x.m_step, 1);
            step = tmp;
        } else {
            step = llvm::ConstantInt::get(context,
                llvm::APInt(32, 1));
        }
        if(!x.m_start && !x.m_end && !x.m_step){
            tmp = str; // no need for slicing
        } else {
            llvm::Value* str_data{}, *str_len{};
            std::tie(str_data, str_len) = llvm_utils->get_string_length_data(ASRUtils::get_string_type(x.m_arg), str);
            tmp = lfortran_str_slice(str_data, str_len, left, right, step, left_present, right_present);
            tmp = llvm_utils->create_string_descriptor(tmp, lfortran_str_len(tmp, false), "stringSection_desc");
        }
    }
    void stringSection_helper_fortran(const ASR::StringSection_t &x){
        this->visit_expr_load_wrapper(x.m_arg, 0);
        llvm::Value *str = tmp;
        llvm::Value *left{}, *right{};
        if (x.m_start) {
            this->visit_expr_load_wrapper(x.m_start, 1, true);
            left = tmp;
        }
        if (x.m_end) {
            this->visit_expr_load_wrapper(x.m_end, 1, true);
            right = tmp;
        }
        LCOMPILERS_ASSERT(x.m_step)
        LCOMPILERS_ASSERT(ASR::is_a<ASR::IntegerConstant_t>(*x.m_step))
        LCOMPILERS_ASSERT(ASR::down_cast<ASR::IntegerConstant_t>(x.m_step)->m_n == 1)
        if(!x.m_start && !x.m_end){
            tmp = str; // no need for slicing
        } else {
            llvm::Value* str_data = llvm_utils->get_string_data(ASRUtils::get_string_type(x.m_arg), str);
            llvm::Value* sliced_str = lfortran_str_slice_fortran(str_data, left, right);
            this->visit_expr_load_wrapper(ASRUtils::get_string_type(x.m_type)->m_len, 1);
            llvm::Value* resulting_length = llvm_utils->convert_kind(tmp, llvm::Type::getInt64Ty(context));
            tmp = nullptr;
            tmp = llvm_utils->create_string_descriptor(sliced_str, resulting_length, "stringSection_desc");
        }
    }

    void visit_StringSection(const ASR::StringSection_t& x) {
        if (x.m_value) {
            this->visit_expr_wrapper(x.m_value, true);
            return;
        }
        // TODO : Find some way to use the helper functions based on the frontend,
        // We are using fortran only for now.
        if(true){ // Fortran 
            stringSection_helper_fortran(x);
        } else { // Python
            stringSection_helper_python(x);
        }
    }

    void visit_StringPhysicalCast(const ASR::StringPhysicalCast_t &x){
        if( x.m_old == ASR::string_physical_typeType::DescriptorString &&
            x.m_new == ASR::string_physical_typeType::CChar){
            // CChar -> is represented as `char*` in LLVM backend.
            this->visit_expr_load_wrapper(x.m_arg, 0);
            tmp = llvm_utils->get_string_data(ASRUtils::get_string_type(x.m_arg), tmp);
        } else if (x.m_old == ASR::string_physical_typeType::CChar &&
            x.m_new == ASR::string_physical_typeType::DescriptorString){
            this->visit_expr_load_wrapper(x.m_arg, 0);// Typically a bind-C-function return 

            int len = -1;
            bool is_const_len = ASRUtils::extract_value(
                ASRUtils::get_string_type(x.m_arg)->m_len, len);
            if(!is_const_len) throw LCompilersException("Unhandled CChar string physical type with non constant length");

            tmp = llvm_utils->create_string_descriptor(tmp,
                llvm::ConstantInt::get(context, llvm::APInt(64, len)),"stringCast_desc");
        } else {
            LCOMPILERS_ASSERT(false);
        }
    }

    void visit_RealCopySign(const ASR::RealCopySign_t& x) {
        if (x.m_value) {
            this->visit_expr_wrapper(x.m_value, true);
            return;
        }
        this->visit_expr(*x.m_target);
        llvm::Value* target = tmp;

        this->visit_expr(*x.m_source);
        llvm::Value* source = tmp;

        llvm::Type *type;
        int a_kind;
        a_kind = down_cast<ASR::Real_t>(ASRUtils::type_get_past_pointer(x.m_type))->m_kind;
        type = llvm_utils->getFPType(a_kind);
        if (ASR::is_a<ASR::ArrayItem_t>(*(x.m_target))) {
            target = llvm_utils->CreateLoad2(type, target);
        }
        if (ASR::is_a<ASR::ArrayItem_t>(*(x.m_source))) {
            source = llvm_utils->CreateLoad2(type, source);
        }
        llvm::Value *ftarget = builder->CreateSIToFP(target,
                type);
        llvm::Value *fsource = builder->CreateSIToFP(source,
                type);
        std::string func_name = a_kind == 4 ? "llvm.copysign.f32" : "llvm.copysign.f64";
        llvm::Function *fn_copysign = module->getFunction(func_name);
        if (!fn_copysign) {
            llvm::FunctionType *function_type = llvm::FunctionType::get(
                    type, { type, type}, false);
            fn_copysign = llvm::Function::Create(function_type,
                    llvm::Function::ExternalLinkage, func_name,
                    module.get());
        }
        tmp = builder->CreateCall(fn_copysign, {ftarget, fsource});
    }

    template <typename T>
    void handle_SU_IntegerBinOp(const T &x, bool signed_int) {
        if (x.m_value) {
            this->visit_expr_wrapper(x.m_value, true);
            return;
        }
        this->visit_expr_load_wrapper(x.m_left, LLVM::is_llvm_pointer(*expr_type(x.m_left)) ? 2 : 1, true);
        llvm::Value *left_val = tmp;
        this->visit_expr_load_wrapper(x.m_right, LLVM::is_llvm_pointer(*expr_type(x.m_right)) ? 2 : 1, true);
        llvm::Value *right_val = tmp;
        LCOMPILERS_ASSERT(ASRUtils::is_integer(*x.m_type) ||
            ASRUtils::is_unsigned_integer(*x.m_type))
        switch (x.m_op) {
            case ASR::binopType::Add: {
                tmp = builder->CreateAdd(left_val, right_val);
                break;
            };
            case ASR::binopType::Sub: {
                tmp = builder->CreateSub(left_val, right_val);
                break;
            };
            case ASR::binopType::Mul: {
                tmp = builder->CreateMul(left_val, right_val);
                break;
            };
            case ASR::binopType::Div: {
                if (signed_int) {
                    tmp = builder->CreateSDiv(left_val, right_val);
                } else {
                    tmp = builder->CreateUDiv(left_val, right_val);
                }
                break;
            };
            case ASR::binopType::Pow: {
                int64_t exponent_const =  INT64_MAX;
                ASRUtils::extract_value(x.m_right, exponent_const);
                // Handle simple-common exponent cases for faster computation.
                if (exponent_const == 2) {
                    tmp = builder->CreateMul(left_val, left_val, "simplified_pow_operation");
                } else if (exponent_const == 3) {
                    tmp = builder->CreateMul(
                            left_val,
                            builder->CreateMul(
                                left_val,
                                left_val,
                                "simplified_pow_operation"),
                            "simplified_pow_operation");
                } else { // Use `pow` function
                    llvm::Type* const i64_ty = llvm::Type::getInt64Ty(context);
                    llvm::Value* _right = llvm_utils->convert_kind(right_val, i64_ty);
                    llvm::Value* _left = llvm_utils->convert_kind(left_val, i64_ty);
                    const std::string func_name = "_lfortran_integer_pow_64";
                    llvm::Function *fn_pow = module->getFunction(func_name);
                    if (!fn_pow) {
                        llvm::FunctionType *function_type = llvm::FunctionType::get(
                                i64_ty, {i64_ty, i64_ty}, false);
                        fn_pow = llvm::Function::Create(function_type,
                                llvm::Function::ExternalLinkage, func_name,
                                module.get());
                    }
                    tmp = builder->CreateCall(fn_pow, {_left, _right});
                    llvm::Type* const return_type =
                        llvm_utils->getIntType(
                            ASRUtils::extract_kind_from_ttype_t(x.m_type)); // returnType of the expression.
                    tmp = llvm_utils->convert_kind(tmp, return_type);
                }
                break;
            };
            case ASR::binopType::BitOr: {
                tmp = builder->CreateOr(left_val, right_val);
                break;
            }
            case ASR::binopType::BitAnd: {
                tmp = builder->CreateAnd(left_val, right_val);
                break;
            }
            case ASR::binopType::BitXor: {
                tmp = builder->CreateXor(left_val, right_val);
                break;
            }
            case ASR::binopType::BitLShift: {
                tmp = builder->CreateShl(left_val, right_val);
                break;
            }
            case ASR::binopType::BitRShift: {
                tmp = builder->CreateAShr(left_val, right_val);
                break;
            }
            case ASR::binopType::LBitRShift: {
                tmp = builder->CreateLShr(left_val, right_val);
                break;
            }
        }
    }

    void visit_IntegerBinOp(const ASR::IntegerBinOp_t &x) {
        handle_SU_IntegerBinOp(x, true);
    }

    void visit_UnsignedIntegerBinOp(const ASR::UnsignedIntegerBinOp_t &x) {
        handle_SU_IntegerBinOp(x, false);
    }

    void visit_RealBinOp(const ASR::RealBinOp_t &x) {
        if (x.m_value) {
            this->visit_expr_wrapper(x.m_value, true);
            return;
        }
        lookup_enum_value_for_nonints = true;
        this->visit_expr_wrapper(x.m_left, true);
        llvm::Value *left_val = tmp;
        this->visit_expr_wrapper(x.m_right, true);
        llvm::Value *right_val = tmp;
        lookup_enum_value_for_nonints = false;
        LCOMPILERS_ASSERT(ASRUtils::is_real(*x.m_type))
        if (ASRUtils::is_simd_array(x.m_right) && is_a<ASR::Var_t>(*x.m_right)) {
            right_val = llvm_utils->CreateLoad(right_val);
        }
        if (ASRUtils::is_simd_array(x.m_left) && is_a<ASR::Var_t>(*x.m_left)) {
            left_val = llvm_utils->CreateLoad(left_val);
        }
        switch (x.m_op) {
            case ASR::binopType::Add: {
                tmp = builder->CreateFAdd(left_val, right_val);
                break;
            };
            case ASR::binopType::Sub: {
                tmp = builder->CreateFSub(left_val, right_val);
                break;
            };
            case ASR::binopType::Mul: {
                tmp = builder->CreateFMul(left_val, right_val);
                break;
            };
            case ASR::binopType::Div: {
                tmp = builder->CreateFDiv(left_val, right_val);
                break;
            };
            case ASR::binopType::Pow: {
                int64_t exponent_const =  INT64_MAX;
                ASRUtils::extract_value(x.m_right, exponent_const);
                // Handle simple-common exponent cases for faster computation.
                if (exponent_const == 2) {
                    tmp = builder->CreateFMul(left_val, left_val, "simplified_pow_operation");
                } else if (exponent_const == 3) {
                    tmp = builder->CreateFMul(
                            left_val,
                            builder->CreateFMul(
                                left_val,
                                left_val,
                                "simplified_pow_operation"),
                            "simplified_pow_operation");
                } else { // Use `pow` function
                    const int return_kind = down_cast<ASR::Real_t>(ASRUtils::extract_type(x.m_type))->m_kind;
                    llvm::Type* const base_type = llvm_utils->getFPType(return_kind);
                    llvm::Type *exponent_type = nullptr;
                    std::string func_name;
                    // Choose the appropriate llvm_pow* intrinsic function + Set the exponent type.
                    if(ASRUtils::is_integer(*ASRUtils::expr_type(x.m_right))) {
                        #if LLVM_VERSION_MAJOR <= 12
                        func_name = (return_kind == 4) ? "llvm.powi.f32" : "llvm.powi.f64";
                        #else
                        func_name = (return_kind == 4) ? "llvm.powi.f32.i32" : "llvm.powi.f64.i32";
                        #endif
                        right_val = llvm_utils->convert_kind(right_val, llvm::Type::getInt32Ty(context)); // `llvm.powi` only has `i32` exponent.
                        exponent_type = llvm::Type::getInt32Ty(context);
                    } else if (ASRUtils::is_real(*ASRUtils::expr_type(x.m_right))) {
                        func_name = (return_kind == 4) ? "llvm.pow.f32" : "llvm.pow.f64";
                        right_val = llvm_utils->convert_kind(right_val, base_type); // `llvm.pow` exponent and base kinds have to match.
                        exponent_type = base_type;
                    } else {
                        LCOMPILERS_ASSERT_MSG(false, "Exponent in RealBinOp should either be [Integer or Real] only.")
                    }

                    llvm::Function *fn_pow = module->getFunction(func_name);
                    if (!fn_pow) {
                        llvm::FunctionType *function_type = llvm::FunctionType::get(
                                base_type, { base_type, exponent_type }, false);
                        fn_pow = llvm::Function::Create(function_type,
                                llvm::Function::ExternalLinkage, func_name,
                                module.get());
                    }
                    tmp = builder->CreateCall(fn_pow, {left_val, right_val});
                }
                break;
            };
            default: {
                throw CodeGenError("Binary operator '" + ASRUtils::binop_to_str_python(x.m_op) + "' not supported",
                    x.base.base.loc);
            }
        }
    }

    void visit_ComplexBinOp(const ASR::ComplexBinOp_t &x) {
        if (x.m_value) {
            this->visit_expr_wrapper(x.m_value, true);
            return;
        }
        this->visit_expr_wrapper(x.m_left, true);
        llvm::Value *left_val = tmp;
        this->visit_expr_wrapper(x.m_right, true);
        llvm::Value *right_val = tmp;
        LCOMPILERS_ASSERT(ASRUtils::is_complex(*x.m_type));
        llvm::Type *type;
        int a_kind;
        a_kind = ASR::down_cast<ASR::Complex_t>(
            ASRUtils::type_get_past_array(
                ASRUtils::type_get_past_pointer(x.m_type)))->m_kind;
        type = llvm_utils->getComplexType(a_kind);
        if( left_val->getType()->isPointerTy() ) {
            left_val = llvm_utils->CreateLoad(left_val);
        }
        if( right_val->getType()->isPointerTy() ) {
            right_val = llvm_utils->CreateLoad(right_val);
        }
        std::string fn_name;
        switch (x.m_op) {
            case ASR::binopType::Add: {
                if (a_kind == 4) {
                    fn_name = "_lfortran_complex_add_32";
                } else {
                    fn_name = "_lfortran_complex_add_64";
                }
                break;
            };
            case ASR::binopType::Sub: {
                if (a_kind == 4) {
                    fn_name = "_lfortran_complex_sub_32";
                } else {
                    fn_name = "_lfortran_complex_sub_64";
                }
                break;
            };
            case ASR::binopType::Mul: {
                if (a_kind == 4) {
                    fn_name = "_lfortran_complex_mul_32";
                } else {
                    fn_name = "_lfortran_complex_mul_64";
                }
                break;
            };
            case ASR::binopType::Div: {
                if (a_kind == 4) {
                    fn_name = "_lfortran_complex_div_32";
                } else {
                    fn_name = "_lfortran_complex_div_64";
                }
                break;
            };
            case ASR::binopType::Pow: {
                if (a_kind == 4) {
                    fn_name = "_lfortran_complex_pow_32";
                } else {
                    fn_name = "_lfortran_complex_pow_64";
                }
                break;
            };
            default: {
                throw CodeGenError("Binary operator '" + ASRUtils::binop_to_str_python(x.m_op) + "' not supported",
                    x.base.base.loc);
            }
        }
        tmp = lfortran_complex_bin_op(left_val, right_val, fn_name, type);
    }

    void visit_OverloadedBinOp(const ASR::OverloadedBinOp_t &x) {
        this->visit_expr(*x.m_overloaded);
    }

    void visit_OverloadedBoolOp(const ASR::OverloadedBoolOp_t& x) {
        this->visit_expr(*x.m_overloaded);
    }

    void visit_OverloadedUnaryMinus(const ASR::OverloadedUnaryMinus_t &x) {
        this->visit_expr(*x.m_overloaded);
    }

    void visit_IntegerBitNot(const ASR::IntegerBitNot_t &x) {
        if (x.m_value) {
            this->visit_expr_wrapper(x.m_value, true);
            return;
        }
        this->visit_expr_wrapper(x.m_arg, true);
        tmp = builder->CreateNot(tmp);
    }

    void visit_UnsignedIntegerBitNot(const ASR::UnsignedIntegerBitNot_t &x) {
        if (x.m_value) {
            this->visit_expr_wrapper(x.m_value, true);
            return;
        }
        this->visit_expr_wrapper(x.m_arg, true);
        tmp = builder->CreateNot(tmp);
    }

    template <typename T>
    void handle_SU_IntegerUnaryMinus(const T& x) {
        if (x.m_value) {
            this->visit_expr_wrapper(x.m_value, true);
            return;
        }
        this->visit_expr_wrapper(x.m_arg, true);
        llvm::Value *zero = llvm::ConstantInt::get(context,
            llvm::APInt(ASRUtils::extract_kind_from_ttype_t(ASRUtils::expr_type(x.m_arg)) * 8, 0));
        tmp = builder->CreateSub(zero, tmp);
    }

    void visit_IntegerUnaryMinus(const ASR::IntegerUnaryMinus_t &x) {
        handle_SU_IntegerUnaryMinus(x);
    }

    void visit_UnsignedIntegerUnaryMinus(const ASR::UnsignedIntegerUnaryMinus_t &x) {
        handle_SU_IntegerUnaryMinus(x);
    }

    void visit_RealUnaryMinus(const ASR::RealUnaryMinus_t &x) {
        if (x.m_value) {
            this->visit_expr_wrapper(x.m_value, true);
            return;
        }
        this->visit_expr_wrapper(x.m_arg, true);
        tmp = builder->CreateFNeg(tmp);
    }

    void visit_ComplexUnaryMinus(const ASR::ComplexUnaryMinus_t &x) {
        if (x.m_value) {
            this->visit_expr_wrapper(x.m_value, true);
            return;
        }
        this->visit_expr_wrapper(x.m_arg, true);
        llvm::Type *type = tmp->getType();
        llvm::Value *re = complex_re(tmp, type);
        llvm::Value *im = complex_im(tmp, type);
        re = builder->CreateFNeg(re);
        im = builder->CreateFNeg(im);
        tmp = complex_from_floats(re, im, type);
    }

    template <typename T>
    void handle_SU_IntegerConstant(const T &x) {
        int64_t val = x.m_n;
        int a_kind = ASRUtils::extract_kind_from_ttype_t(x.m_type);
        switch( a_kind ) {
            case 1: {
                tmp = llvm::ConstantInt::get(context, llvm::APInt(8, static_cast<int8_t>(val), true));
                break ;
            }
            case 2: {
                tmp = llvm::ConstantInt::get(context, llvm::APInt(16, static_cast<int16_t>(val), true));
                break ;
            }
            case 4 : {
                tmp = llvm::ConstantInt::get(context, llvm::APInt(32, static_cast<int32_t>(val), true));
                break;
            }
            case 8 : {
                tmp = llvm::ConstantInt::get(context, llvm::APInt(64, val, true));
                break;
            }
            default : {
                throw CodeGenError("Constant integers of " + std::to_string(a_kind)
                                    + " bytes aren't supported yet.");
            }

        }
    }

    void visit_IntegerConstant(const ASR::IntegerConstant_t &x) {
        handle_SU_IntegerConstant(x);
    }

    void visit_UnsignedIntegerConstant(const ASR::UnsignedIntegerConstant_t &x) {
        handle_SU_IntegerConstant(x);
    }

    void visit_RealConstant(const ASR::RealConstant_t &x) {
        double val = x.m_r;
        int a_kind = ((ASR::Real_t*)(&(x.m_type->base)))->m_kind;
        switch( a_kind ) {

            case 4 : {
                tmp = llvm::ConstantFP::get(context, llvm::APFloat((float)val));
                break;
            }
            case 8 : {
                tmp = llvm::ConstantFP::get(context, llvm::APFloat(val));
                break;
            }
            default : {
                break;
            }

        }

    }

    template <typename T>
    void visit_ArrayConstructorUtil(const T& x) {
        if (x.m_value) {
            this->visit_expr_wrapper(x.m_value, true);
            return;
        }

        llvm::Type* el_type = nullptr;
        ASR::ttype_t* x_m_type = ASRUtils::type_get_past_array(x.m_type);
        if (ASR::is_a<ASR::Integer_t>(*x_m_type)) {
            el_type = llvm_utils->getIntType(ASR::down_cast<ASR::Integer_t>(x_m_type)->m_kind);
        } else if (ASR::is_a<ASR::Real_t>(*x_m_type)) {
            switch (ASR::down_cast<ASR::Real_t>(x_m_type)->m_kind) {
                case (4) :
                    el_type = llvm::Type::getFloatTy(context); break;
                case (8) :
                    el_type = llvm::Type::getDoubleTy(context); break;
                default :
                    throw CodeGenError("ConstArray real kind not supported yet");
            }
        } else if (ASR::is_a<ASR::Logical_t>(*x_m_type)) {
            el_type = llvm::Type::getInt1Ty(context);
        } else if (ASR::is_a<ASR::String_t>(*x_m_type)) {
            el_type = character_type;
        } else if (ASR::is_a<ASR::Complex_t>(*x_m_type)) {
            int complex_kind = ASR::down_cast<ASR::Complex_t>(x_m_type)->m_kind;
            if( complex_kind == 4 ) {
                el_type = llvm_utils->complex_type_4;
            } else if( complex_kind == 8 ) {
                el_type = llvm_utils->complex_type_8;
            } else {
                LCOMPILERS_ASSERT(false);
            }
        } else {
            throw CodeGenError("ConstArray type not supported yet");
        }
        // Create <n x float> type, where `n` is the length of the `x` constant array
        llvm::Type* type_fxn = FIXED_VECTOR_TYPE::get(el_type, ASRUtils::get_fixed_size_of_array(x.m_type));
        // Create a pointer <n x float>* to a stack allocated <n x float>
        llvm::AllocaInst *p_fxn = llvm_utils->CreateAlloca(*builder, type_fxn);
        // Assign the array elements to `p_fxn`.
        for (size_t i=0; i < x.n_args; i++) {
            llvm::Value *llvm_el = llvm_utils->create_gep2(type_fxn, p_fxn, i);
            ASR::expr_t *el = x.m_args[i];
            int64_t ptr_loads_copy = ptr_loads;
            ptr_loads = 2;
            this->visit_expr_wrapper(el, true);
            ptr_loads = ptr_loads_copy;
            builder->CreateStore(tmp, llvm_el);
        }
        // Return the vector as float* type:
        tmp = llvm_utils->create_gep2(type_fxn ,p_fxn, 0);
    }

    void visit_ArrayConstantUtil(const ASR::ArrayConstant_t &x) {
        llvm::Type* el_type = nullptr;
        ASR::ttype_t* x_m_type = ASRUtils::type_get_past_array(x.m_type);
        if (ASR::is_a<ASR::Integer_t>(*x_m_type)) {
            el_type = llvm_utils->getIntType(ASR::down_cast<ASR::Integer_t>(x_m_type)->m_kind);
        } else if (ASR::is_a<ASR::Real_t>(*x_m_type)) {
            switch (ASR::down_cast<ASR::Real_t>(x_m_type)->m_kind) {
                case (4) :
                    el_type = llvm::Type::getFloatTy(context); break;
                case (8) :
                    el_type = llvm::Type::getDoubleTy(context); break;
                default :
                    throw CodeGenError("ConstArray real kind not supported yet");
            }
        } else if (ASR::is_a<ASR::Logical_t>(*x_m_type)) {
            el_type = llvm::Type::getInt1Ty(context);
        } else if (ASR::is_a<ASR::String_t>(*x_m_type)) {
            el_type = llvm_utils->get_StringType(x_m_type);
        } else if (ASR::is_a<ASR::Complex_t>(*x_m_type)) {
            int complex_kind = ASR::down_cast<ASR::Complex_t>(x_m_type)->m_kind;
            if( complex_kind == 4 ) {
                el_type = llvm_utils->complex_type_4;
            } else if( complex_kind == 8 ) {
                el_type = llvm_utils->complex_type_8;
            } else {
                LCOMPILERS_ASSERT(false);
            }
        } else {
            throw CodeGenError("ConstArray type not supported yet");
        }

        // Declaring array constant as global constant and directly using it
        // instead of storing each element using CreateStore
        int64_t arr_size = ASRUtils::get_fixed_size_of_array(x.m_type);
        llvm::Type *Int32Ty = llvm::Type::getInt32Ty(context);
        llvm::ArrayType * arr_type = llvm::ArrayType::get(el_type, arr_size);
        std::vector<llvm::Constant *> values;

        if (ASRUtils::is_integer(*x_m_type)) {
            for (size_t i=0; i < (size_t) arr_size; i++) {
                ASR::expr_t *el = ASRUtils::fetch_ArrayConstant_value(al, x, i);
                values.push_back(llvm::ConstantInt::get(el_type, down_cast<ASR::IntegerConstant_t>(el)->m_n));
            }
        } else if (ASRUtils::is_real(*x_m_type)) {
            for (size_t i=0; i < (size_t) arr_size; i++) {
                ASR::expr_t *el = ASRUtils::fetch_ArrayConstant_value(al, x, i);
                values.push_back(llvm::ConstantFP::get(el_type, down_cast<ASR::RealConstant_t>(el)->m_r));
            }
        } else if (ASRUtils::is_logical(*x_m_type)) {
            for (size_t i=0; i < (size_t) arr_size; i++) {
                ASR::expr_t *el = ASRUtils::fetch_ArrayConstant_value(al, x, i);
                values.push_back(llvm::ConstantInt::get(el_type, down_cast<ASR::LogicalConstant_t>(el)->m_value));
            }
        } else if (ASRUtils::is_complex(*x_m_type)) {
            for (size_t i=0; i < (size_t) arr_size; i++) {
                ASR::expr_t *el = ASRUtils::fetch_ArrayConstant_value(al, x, i);
                ASR::ComplexConstant_t *comp_const = down_cast<ASR::ComplexConstant_t>(el);
                if (ASRUtils::extract_kind_from_ttype_t(comp_const->m_type) == 4) {
                    values.push_back(llvm::ConstantStruct::get(llvm_utils->complex_type_4,
                        {llvm::ConstantFP::get(llvm::Type::getFloatTy(context), comp_const->m_re),
                        llvm::ConstantFP::get(llvm::Type::getFloatTy(context), comp_const->m_im)}));
                } else {
                    values.push_back(llvm::ConstantStruct::get(llvm_utils->complex_type_8,
                        {llvm::ConstantFP::get(llvm::Type::getDoubleTy(context), comp_const->m_re),
                        llvm::ConstantFP::get(llvm::Type::getDoubleTy(context), comp_const->m_im)}));
                }
            }
        } else if (ASRUtils::is_character(*x_m_type)) { // Sepcial Case.
            tmp = llvm_utils->declare_constant_stringArray(al, &x);
            return;
        }

        llvm::Constant *ConstArray = llvm::ConstantArray::get(arr_type, values);
        llvm::GlobalVariable *global_var = new llvm::GlobalVariable(*module, arr_type, true,
            llvm::GlobalValue::PrivateLinkage, ConstArray, "global_array_" + std::to_string(global_array_count++));
        tmp = builder->CreateGEP(
            arr_type, global_var, {llvm::ConstantInt::get(Int32Ty, 0), llvm::ConstantInt::get(Int32Ty, 0)});
    }

    void visit_ArrayConstructor(const ASR::ArrayConstructor_t &x) {
        visit_ArrayConstructorUtil(x);
    }

    void visit_ArrayConstant(const ASR::ArrayConstant_t &x) {
        visit_ArrayConstantUtil(x);
    }

    void visit_Assert(const ASR::Assert_t &x) {
        if (compiler_options.emit_debug_info) debug_emit_loc(x);
        this->visit_expr_wrapper(x.m_test, true);
        llvm_utils->create_if_else(tmp, []() {}, [=]() {
            if (compiler_options.emit_debug_info) {
                llvm::Value *fmt_ptr = builder->CreateGlobalStringPtr(infile);
                llvm::Value *fmt_ptr1 = llvm::ConstantInt::get(context, llvm::APInt(
                    1, compiler_options.use_colors));
                call_print_stacktrace_addresses(context, *module, *builder,
                    {fmt_ptr, fmt_ptr1});
            }
            if (x.m_msg) {
                std::vector<std::string> fmt;
                std::vector<llvm::Value *> args;
                fmt.push_back("%s");
                args.push_back(builder->CreateGlobalStringPtr("AssertionError: "));
                compute_fmt_specifier_and_arg(fmt, args, x.m_msg, x.base.base.loc);
                fmt.push_back("%s");
                args.push_back(builder->CreateGlobalStringPtr("\n"));
                std::string fmt_str;
                for (size_t i=0; i<fmt.size(); i++) {
                    fmt_str += fmt[i];
                }
                llvm::Value *fmt_ptr = builder->CreateGlobalStringPtr(fmt_str);
                std::vector<llvm::Value *> print_error_args;
                print_error_args.push_back(fmt_ptr);
                print_error_args.insert(print_error_args.end(), args.begin(), args.end());
                print_error(context, *module, *builder, print_error_args);
            } else {
                llvm::Value *fmt_ptr = builder->CreateGlobalStringPtr("AssertionError\n");
                print_error(context, *module, *builder, {fmt_ptr});
            }
            int exit_code_int = 1;
            llvm::Value *exit_code = llvm::ConstantInt::get(context,
                    llvm::APInt(32, exit_code_int));
            exit(context, *module, *builder, exit_code);
        });
    }

    void visit_ComplexConstructor(const ASR::ComplexConstructor_t &x) {
        if (x.m_value) {
            this->visit_expr_wrapper(x.m_value, true);
            return;
        }
        this->visit_expr_wrapper(x.m_re, true);
        llvm::Value *re_val = tmp;

        this->visit_expr_wrapper(x.m_im, true);
        llvm::Value *im_val = tmp;

        int a_kind = ASRUtils::extract_kind_from_ttype_t(x.m_type);

        llvm::Value *re2, *im2;
        llvm::Type *type;
        switch( a_kind ) {
            case 4: {
                re2 = builder->CreateFPTrunc(re_val, llvm::Type::getFloatTy(context));
                im2 = builder->CreateFPTrunc(im_val, llvm::Type::getFloatTy(context));
                type = complex_type_4;
                break;
            }
            case 8: {
                re2 = builder->CreateFPExt(re_val, llvm::Type::getDoubleTy(context));
                im2 = builder->CreateFPExt(im_val, llvm::Type::getDoubleTy(context));
                type = complex_type_8;
                break;
            }
            default: {
                throw CodeGenError("kind type is not supported");
            }
        }
        tmp = complex_from_floats(re2, im2, type);
    }

    void visit_ComplexConstant(const ASR::ComplexConstant_t &x) {
        double re = x.m_re;
        double im = x.m_im;
        int a_kind = ASRUtils::extract_kind_from_ttype_t(x.m_type);
        llvm::Value *re2, *im2;
        llvm::Type *type;
        switch( a_kind ) {
            case 4: {
                re2 = llvm::ConstantFP::get(context, llvm::APFloat((float)re));
                im2 = llvm::ConstantFP::get(context, llvm::APFloat((float)im));
                type = complex_type_4;
                break;
            }
            case 8: {
                re2 = llvm::ConstantFP::get(context, llvm::APFloat(re));
                im2 = llvm::ConstantFP::get(context, llvm::APFloat(im));
                type = complex_type_8;
                break;
            }
            default: {
                throw CodeGenError("kind type is not supported");
            }
        }
        tmp = complex_from_floats(re2, im2, type);
    }

    void visit_LogicalConstant(const ASR::LogicalConstant_t &x) {
        int val;
        if (x.m_value == true) {
            val = 1;
        } else {
            val = 0;
        }
        tmp = llvm::ConstantInt::get(context, llvm::APInt(1, val));
    }

    void visit_LogicalNot(const ASR::LogicalNot_t &x) {
        if (x.m_value) {
            this->visit_expr_wrapper(x.m_value, true);
            return;
        }
        this->visit_expr_wrapper(x.m_arg, true);
        llvm::Value *arg = tmp;
        tmp = builder->CreateNot(arg);
    }

    void visit_StringConstant(const ASR::StringConstant_t &x) {
        tmp = llvm_utils->declare_string_constant(&x);
    }

    inline void fetch_ptr(ASR::Variable_t* x) {
        uint32_t x_h = get_hash((ASR::asr_t*)x);
        LCOMPILERS_ASSERT(llvm_symtab.find(x_h) != llvm_symtab.end());
        llvm::Value* x_v = llvm_symtab[x_h];
        int64_t ptr_loads_copy = ptr_loads;
        tmp = x_v;
#if LLVM_VERSION_MAJOR > 16
        int loads = 0;
#endif
        while( ptr_loads_copy-- ) {
#if LLVM_VERSION_MAJOR > 16
            if( loads == 0 ) {
                ptr_type[tmp] = llvm_utils->get_type_from_ttype_t_util(ASRUtils::EXPR(ASR::make_Var_t(
                    al, x->base.base.loc, &x->base)),
                    x->m_type, module.get());
            } else {
                ptr_type[tmp] = llvm_utils->get_type_from_ttype_t_util(ASRUtils::EXPR(ASR::make_Var_t(
                    al, x->base.base.loc, &x->base)),
                    ASRUtils::type_get_past_allocatable_pointer(x->m_type),
                    module.get());
            }
#endif
            tmp = llvm_utils->CreateLoad(tmp);
#if LLVM_VERSION_MAJOR > 16
            loads++;
#endif
        }
    }

    inline void fetch_val(ASR::Variable_t* x) {
        uint32_t x_h = get_hash((ASR::asr_t*)x);
        llvm::Value* x_v;
        LCOMPILERS_ASSERT(llvm_symtab.find(x_h) != llvm_symtab.end());
        x_v = llvm_symtab[x_h];
        if (x->m_value_attr) {
            // Already a value, such as value argument to bind(c)
            tmp = x_v;
            return;
        }
        if( ASRUtils::is_array(x->m_type) ) {
            tmp = x_v;
        } else {
            tmp = x_v;
            // Load only once since its a value
            if( ptr_loads > 0 ) {
                llvm::Type* x_llvm_type = llvm_utils->get_type_from_ttype_t_util(ASRUtils::EXPR(ASR::make_Var_t(
                    al, x->base.base.loc, &x->base)), x->m_type, module.get());
                tmp = llvm_utils->CreateLoad2(x_llvm_type, tmp);
            }
        }
    }

    inline void fetch_var(ASR::Variable_t* x) {
        // Only do for constant variables
        if (x->m_value && x->m_storage == ASR::storage_typeType::Parameter) {
            this->visit_expr_wrapper(x->m_value, true);
            return;
        }
        ASR::ttype_t *t2_ = ASRUtils::type_get_past_array(x->m_type);
        switch( t2_->type ) {
            case ASR::ttypeType::Pointer:
            case ASR::ttypeType::Allocatable: {
                ASR::ttype_t *t2 = ASRUtils::extract_type(x->m_type);
                switch (t2->type) {
                    case ASR::ttypeType::Integer:
                    case ASR::ttypeType::UnsignedInteger:
                    case ASR::ttypeType::Real:
                    case ASR::ttypeType::Complex:
                    case ASR::ttypeType::StructType:
                    case ASR::ttypeType::String:
                    case ASR::ttypeType::Logical:
                    case ASR::ttypeType::CPtr:{
                        if( t2->type == ASR::ttypeType::StructType ) {
                            current_der_type_name = ASRUtils::symbol_name(
                                ASRUtils::symbol_get_past_external(ASRUtils::get_struct_sym_from_struct_expr(ASRUtils::EXPR(ASR::make_Var_t(al, x->base.base.loc, (ASR::symbol_t*) x)))));
                        }
                        if (ASRUtils::is_unlimited_polymorphic_type(ASRUtils::EXPR(ASR::make_Var_t(al, x->base.base.loc, (ASR::symbol_t*)x)))) {
                            uint32_t h = get_hash((ASR::asr_t*)x);
                            if( llvm_symtab.find(h) != llvm_symtab.end() ) {
                                tmp = llvm_symtab[h];
                            }
                        } else {
                            fetch_ptr(x);
                        }
                        break;
                    }
                    default:
                        break;
                }
                break;
            }
            case ASR::ttypeType::StructType: {
                ASR::Struct_t* der_type = ASR::down_cast<ASR::Struct_t>(ASRUtils::symbol_get_past_external(x->m_type_declaration));
                current_der_type_name = std::string(der_type->m_name);
                uint32_t h = get_hash((ASR::asr_t*)x);
                if( llvm_symtab.find(h) != llvm_symtab.end() ) {
                    tmp = llvm_symtab[h];
                }
                break;
            }
            case ASR::ttypeType::UnionType: {
                ASR::UnionType_t* der = ASR::down_cast<ASR::UnionType_t>(t2_);
                ASR::Union_t* der_type = ASR::down_cast<ASR::Union_t>(
                    ASRUtils::symbol_get_past_external(der->m_union_type));
                current_der_type_name = std::string(der_type->m_name);
                uint32_t h = get_hash((ASR::asr_t*)x);
                if( llvm_symtab.find(h) != llvm_symtab.end() ) {
                    tmp = llvm_symtab[h];
                }
                break;
            }
            case ASR::ttypeType::FunctionType: {
                // break;
                uint32_t h = get_hash((ASR::asr_t*)x);
                uint32_t x_h = get_hash((ASR::asr_t*)x);
                if ( llvm_symtab_fn_arg.find(h) != llvm_symtab_fn_arg.end() ) {
                    tmp = llvm_symtab_fn_arg[h];
                } else if ( llvm_symtab.find(x_h) != llvm_symtab.end() ) {
                    tmp = llvm_symtab[x_h];
                } else if (llvm_symtab_fn.find(h) != llvm_symtab_fn.end()) {
                    tmp = llvm_symtab_fn[h];
                    tmp = llvm_utils->CreateLoad2(tmp->getType()->getPointerTo(), tmp);
#if LLVM_VERSION_MAJOR > 16
                    ptr_type[tmp] = tmp->getType();
#endif
                } else {
                    throw CodeGenError("Function type not supported yet");
                }
                if (x->m_value_attr) {
                    // Already a value, such as value argument to bind(c)
                    break;
                }
                if( ASRUtils::is_array(x->m_type) ) {
                    break;
                } else {
                    // Load only once since its a value
                    if( ptr_loads > 0 ) {
                        llvm::Type* x_llvm_type = llvm_utils->get_type_from_ttype_t_util(ASRUtils::EXPR(ASR::make_Var_t(
                            al, x->base.base.loc, &x->base)), x->m_type, module.get());
                        tmp = llvm_utils->CreateLoad2(x_llvm_type, tmp);
#if LLVM_VERSION_MAJOR > 16
                    ptr_type[tmp] = tmp->getType();
#endif
                    }
                }
                break;
            } default: {
                fetch_val(x);
                break;
            }
        }
    }

    void visit_Var(const ASR::Var_t &x) {
        ASR::symbol_t* x_m_v = ASRUtils::symbol_get_past_external(x.m_v);
        switch( x_m_v->type ) {
            case ASR::symbolType::Variable: {
                ASR::Variable_t *v = ASR::down_cast<ASR::Variable_t>(x_m_v);
                fetch_var(v);
                return ;
            }
            case ASR::symbolType::Function: {
                uint32_t h = get_hash((ASR::asr_t*)x_m_v);
                if( llvm_symtab_fn.find(h) != llvm_symtab_fn.end() ) {
                    tmp = llvm_symtab_fn[h];
                }
                return;
            }
            default: {
                throw CodeGenError("Only function and variables supported so far");
            }
        }
    }

    inline ASR::ttype_t* extract_ttype_t_from_expr(ASR::expr_t* expr) {
        return ASRUtils::expr_type(expr);
    }

    void extract_kinds(const ASR::Cast_t& x,
                       int& arg_kind, int& dest_kind)
    {
        dest_kind = ASRUtils::extract_kind_from_ttype_t(x.m_type);
        ASR::ttype_t* curr_type = nullptr;
        if (ASRUtils::is_unlimited_polymorphic_type(x.m_arg)) {
            curr_type = current_select_type_block_type_asr;
        } else {
            curr_type = extract_ttype_t_from_expr(x.m_arg);
        }
        LCOMPILERS_ASSERT(curr_type != nullptr)
        arg_kind = ASRUtils::extract_kind_from_ttype_t(curr_type);
    }

    template <typename T>
    void handle_arr_for_complex_im_re(const T& t) {
        int64_t ptr_loads_copy = ptr_loads;
        ptr_loads = 2 - LLVM::is_llvm_pointer(*ASRUtils::expr_type(t.m_arg));
        this->visit_expr_wrapper(t.m_arg, false);
        ptr_loads = ptr_loads_copy;
        llvm::Value* des_complex_arr = tmp;
        llvm::Type* des_complex_type = llvm_utils->get_type_from_ttype_t_util(t.m_arg,
            ASRUtils::extract_type(ASRUtils::expr_type(t.m_arg)), module.get());
        tmp = llvm_utils->CreateLoad2(des_complex_type->getPointerTo(), arr_descr->get_pointer_to_data(des_complex_arr));
        int kind = ASRUtils::extract_kind_from_ttype_t(t.m_type);
        llvm::Type* pointer_cast_type = nullptr;
        if (kind == 4) {
            pointer_cast_type = llvm::Type::getFloatTy(context)->getPointerTo();
        } else {
            pointer_cast_type = llvm::Type::getDoubleTy(context)->getPointerTo();
        }
        tmp = builder->CreateBitCast(tmp, pointer_cast_type);
        PointerToData_to_Descriptor(t.m_arg, t.m_type, t.m_type);
        llvm::Value* des_real_arr = tmp;
        llvm::Value* arr_data = llvm_utils->CreateLoad2(
            des_complex_type->getPointerTo(), arr_descr->get_pointer_to_data(des_complex_arr));
        tmp = builder->CreateBitCast(arr_data, pointer_cast_type);
        builder->CreateStore(tmp, arr_descr->get_pointer_to_data(des_real_arr));
        if (std::is_same<T, ASR::ComplexIm_t>::value) {
            llvm::Value* incremented_offset = builder->CreateAdd(
                arr_descr->get_offset(des_real_arr, true),
                llvm::ConstantInt::get(context, llvm::APInt(32, 1)));
            builder->CreateStore(incremented_offset, arr_descr->get_offset(des_real_arr, false));
        }
        int n_dims = ASRUtils::extract_n_dims_from_ttype(t.m_type);
        llvm::Value* dim_des_real_arr = arr_descr->get_pointer_to_dimension_descriptor_array(des_real_arr, true);
        for (int i = 0; i < n_dims; i++) {
            llvm::Value* dim_idx = llvm::ConstantInt::get(context, llvm::APInt(32, i));
            llvm::Value* dim_des_real_arr_idx = arr_descr->get_pointer_to_dimension_descriptor(dim_des_real_arr, dim_idx);
            llvm::Value* doubled_stride = builder->CreateMul(
                arr_descr->get_stride(dim_des_real_arr_idx, true),
                llvm::ConstantInt::get(context, llvm::APInt(32, 2)));
            builder->CreateStore(doubled_stride, arr_descr->get_stride(dim_des_real_arr_idx, false));
        }
        tmp = des_real_arr;
    }

    void visit_ComplexRe(const ASR::ComplexRe_t &x) {
        if (x.m_value) {
            this->visit_expr_wrapper(x.m_value, true);
            return;
        }
        if (ASRUtils::is_array(x.m_type)) {
            handle_arr_for_complex_im_re(x);
            return;
        }
        this->visit_expr_wrapper(x.m_arg, true);
        ASR::ttype_t* curr_type = extract_ttype_t_from_expr(x.m_arg);
        int arg_kind = ASRUtils::extract_kind_from_ttype_t(curr_type);
        int dest_kind = ASRUtils::extract_kind_from_ttype_t(x.m_type);
        llvm::Value *re;
        if (arg_kind == 4 && dest_kind == 4) {
            // complex(4) -> real(4)
            re = complex_re(tmp, complex_type_4);
            tmp = re;
        } else if (arg_kind == 4 && dest_kind == 8) {
            // complex(4) -> real(8)
            re = complex_re(tmp, complex_type_4);
            tmp = builder->CreateFPExt(re, llvm::Type::getDoubleTy(context));
        } else if (arg_kind == 8 && dest_kind == 4) {
            // complex(8) -> real(4)
            re = complex_re(tmp, complex_type_8);
            tmp = builder->CreateFPTrunc(re, llvm::Type::getFloatTy(context));
        } else if (arg_kind == 8 && dest_kind == 8) {
            // complex(8) -> real(8)
            re = complex_re(tmp, complex_type_8);
            tmp = re;
        } else {
            std::string msg = "Conversion from " + std::to_string(arg_kind) +
                              " to " + std::to_string(dest_kind) + " not implemented yet.";
            throw CodeGenError(msg);
        }
    }

    void visit_ComplexIm(const ASR::ComplexIm_t &x) {
        if (x.m_value) {
            this->visit_expr_wrapper(x.m_value, true);
            return;
        }
        if (ASRUtils::is_array(x.m_type)) {
            handle_arr_for_complex_im_re(x);
            return;
        }
        ASR::ttype_t* curr_type = extract_ttype_t_from_expr(x.m_arg);
        int arg_kind = ASRUtils::extract_kind_from_ttype_t(curr_type);
        llvm::Function *fn = nullptr;
        llvm::Type *ret_type = nullptr, *complex_type = nullptr;
        llvm::AllocaInst *arg = nullptr;
        std::string runtime_func_name = "";
        if (arg_kind == 4) {
            runtime_func_name = "_lfortran_complex_aimag_32";
            ret_type = llvm::Type::getFloatTy(context);
            complex_type = complex_type_4;
            arg = llvm_utils->CreateAlloca(*builder, complex_type_4,
                nullptr);
        } else {
            runtime_func_name = "_lfortran_complex_aimag_64";
            ret_type = llvm::Type::getDoubleTy(context);
            complex_type = complex_type_8;
            arg = llvm_utils->CreateAlloca(*builder, complex_type_8);
        }
        fn = module->getFunction(runtime_func_name);
        if (!fn) {
            llvm::FunctionType *function_type = llvm::FunctionType::get(
                    llvm::Type::getVoidTy(context), {
                        complex_type->getPointerTo(),
                        ret_type->getPointerTo(),
                    }, false);
            fn = llvm::Function::Create(function_type,
                    llvm::Function::ExternalLinkage, runtime_func_name, *module);
        }
        this->visit_expr_wrapper(x.m_arg, true);
        builder->CreateStore(tmp, arg);
        llvm::AllocaInst *result = llvm_utils->CreateAlloca(*builder, ret_type);
        std::vector<llvm::Value*> args = {arg, result};
        builder->CreateCall(fn, args);
        tmp = llvm_utils->CreateLoad2(ret_type, result);
    }

    void visit_BitCast(const ASR::BitCast_t& x) {
        if (x.m_value) {
            this->visit_expr_wrapper(x.m_value, true);
            return;
        }
        llvm::Value* source{};
        this->visit_expr_load_wrapper(x.m_source,
            ASRUtils::is_character(*expr_type(x.m_source)) ? 0 : ptr_loads,
            true);
        source = tmp;
        llvm::Type* source_type = llvm_utils->get_type_from_ttype_t_util(x.m_source, ASRUtils::expr_type(x.m_source), module.get());
        llvm::Value* source_ptr;
        bool is_array = ASRUtils::is_array(ASRUtils::expr_type(x.m_source));
        if (ASRUtils::is_character(*expr_type(x.m_source))) {
            source_ptr = ASRUtils::is_array_of_strings(expr_type(x.m_source)) ? 
                        llvm_utils->get_stringArray_data(expr_type(x.m_source), tmp) :
                        llvm_utils->get_string_data(ASRUtils::get_string_type(x.m_source), tmp);
        } else if(source->getType()->isPointerTy() || source_type->isArrayTy()){//Case: [n x i8]* type Arrays and ptr %
            source_ptr = source;
        } else if (is_array) {
            source_type = source->getType();
            source_ptr = llvm_utils->CreateAlloca(source_type, nullptr, "bitcast_source");
            builder->CreateStore(source, source_ptr);
            ASR::Array_t* arr = ASR::down_cast<ASR::Array_t>(
                ASRUtils::type_get_past_allocatable_pointer(ASRUtils::expr_type(x.m_source)));
            llvm::Type* source_type_ = llvm_utils->get_type_from_ttype_t_util(x.m_source,
                ASRUtils::extract_type(ASRUtils::expr_type(x.m_source)), module.get());
            llvm::Type* llvm_source_type_ = llvm_utils->get_type_from_ttype_t_util(x.m_source,
                ASRUtils::type_get_past_allocatable_pointer(ASRUtils::expr_type(x.m_source)), module.get());
            if (arr->m_physical_type == ASR::array_physical_typeType::DescriptorArray) {
                source_ptr = llvm_utils->create_gep2(llvm_source_type_, source_ptr, 0);
                source_ptr = builder->CreateLoad(source_type_->getPointerTo(), source_ptr);
            } else {      // For PointerToDataArray source itself is a pointer to data
                source_ptr = source;
            }
        } else {
            source_ptr = llvm_utils->CreateAlloca(source_type, nullptr, "bitcast_source");
            builder->CreateStore(source, source_ptr);
        }
        llvm::Type* target_base_type = llvm_utils->get_type_from_ttype_t_util(const_cast<ASR::expr_t*>(&x.base), ASRUtils::type_get_past_array(x.m_type), module.get());
        if (ASRUtils::is_character(*expr_type(x.m_source))) {
            tmp = builder->CreateBitCast(source_ptr, target_base_type->getPointerTo());
        } else {
            llvm::Type* target_llvm_type = target_base_type->getPointerTo();
            if ( !ASRUtils::types_equal(ASRUtils::extract_type(ASRUtils::expr_type(x.m_source)), ASRUtils::extract_type(x.m_type), false) &&
                 !( ASR::is_a<ASR::String_t>(*ASRUtils::extract_type(ASRUtils::expr_type(x.m_source))) && ASRUtils::is_integer(*ASRUtils::extract_type(x.m_type)) &&
                   ASR::down_cast<ASR::Integer_t>(ASRUtils::extract_type(x.m_type))->m_kind == 1 ) ) {
                tmp = llvm_utils->CreateLoad2(target_base_type, builder->CreateBitCast(source_ptr, target_llvm_type));
            }
        }
    }

    llvm::Value* get_pointer_to_variable(ASR::expr_t* var) {
        if (llvm_symtab.find((uint64_t)var) != llvm_symtab.end()) {
            return llvm_symtab[(uint64_t)var];
        }
        this->visit_expr_wrapper(var, true);
        llvm::Value *val = tmp;
        llvm::AllocaInst *alloc = builder->CreateAlloca(val->getType(), nullptr);
        builder->CreateStore(val, alloc);
        llvm_symtab[(uint64_t)var] = alloc;
        return alloc;
    }

    void visit_Cast(const ASR::Cast_t &x) {
        if (x.m_value) {
            this->visit_expr_wrapper(x.m_value, true);
            return;
        }
        int64_t ptr_loads_copy = ptr_loads;
        if (ASRUtils::is_pointer(ASRUtils::expr_type(x.m_arg))) {
            ptr_loads = 2;
        }
        this->visit_expr_wrapper(x.m_arg, true);
        switch (x.m_kind) {
            case (ASR::cast_kindType::IntegerToReal) : {
                int a_kind = ASRUtils::extract_kind_from_ttype_t(x.m_type);
                tmp = builder->CreateSIToFP(tmp, llvm_utils->getFPType(a_kind, false));
                break;
            }
            case (ASR::cast_kindType::UnsignedIntegerToReal) : {
                int a_kind = ASRUtils::extract_kind_from_ttype_t(x.m_type);
                tmp = builder->CreateSIToFP(tmp, llvm_utils->getFPType(a_kind, false));
                break;
            }
            case (ASR::cast_kindType::LogicalToReal) : {
                int a_kind = ASRUtils::extract_kind_from_ttype_t(x.m_type);
                tmp = builder->CreateUIToFP(tmp, llvm_utils->getFPType(a_kind, false));
                break;
            }
            case (ASR::cast_kindType::RealToInteger) : {
                llvm::Type *target_type;
                int a_kind = ASRUtils::extract_kind_from_ttype_t(x.m_type);
                target_type = llvm_utils->getIntType(a_kind);
                tmp = builder->CreateFPToSI(tmp, target_type);
                break;
            }
            case (ASR::cast_kindType::RealToUnsignedInteger) : {
                llvm::Type *target_type;
                int a_kind = ASRUtils::extract_kind_from_ttype_t(x.m_type);
                target_type = llvm_utils->getIntType(a_kind);
                tmp = builder->CreateFPToSI(tmp, target_type);
                break;
            }
            case (ASR::cast_kindType::RealToComplex) : {
                llvm::Type *target_type;
                llvm::Value *zero;
                int a_kind = ASRUtils::extract_kind_from_ttype_t(x.m_type);
                switch(a_kind)
                {
                    case 4:
                        target_type = complex_type_4;
                        tmp = builder->CreateFPTrunc(tmp, llvm::Type::getFloatTy(context));
                        zero = llvm::ConstantFP::get(context, llvm::APFloat((float)0.0));
                        break;
                    case 8:
                        target_type = complex_type_8;
                        tmp = builder->CreateFPExt(tmp, llvm::Type::getDoubleTy(context));
                        zero = llvm::ConstantFP::get(context, llvm::APFloat(0.0));
                        break;
                    default:
                        throw CodeGenError("Only 32 and 64 bits real kinds are supported.");
                }
                tmp = complex_from_floats(tmp, zero, target_type);
                break;
            }
            case (ASR::cast_kindType::IntegerToComplex) : {
                int a_kind = ASRUtils::extract_kind_from_ttype_t(x.m_type);
                llvm::Type *target_type;
                llvm::Type *complex_type;
                llvm::Value *zero;
                switch(a_kind)
                {
                    case 4:
                        target_type = llvm::Type::getFloatTy(context);
                        complex_type = complex_type_4;
                        zero = llvm::ConstantFP::get(context, llvm::APFloat((float)0.0));
                        break;
                    case 8:
                        target_type = llvm::Type::getDoubleTy(context);
                        complex_type = complex_type_8;
                        zero = llvm::ConstantFP::get(context, llvm::APFloat(0.0));
                        break;
                    default:
                        throw CodeGenError("Only 32 and 64 bits real kinds are supported.");
                }
                tmp = builder->CreateSIToFP(tmp, target_type);
                tmp = complex_from_floats(tmp, zero, complex_type);
                break;
            }
            case (ASR::cast_kindType::IntegerToLogical) :
            case (ASR::cast_kindType::UnsignedIntegerToLogical) : {
                ASR::ttype_t* curr_type = extract_ttype_t_from_expr(x.m_arg);
                LCOMPILERS_ASSERT(curr_type != nullptr)
                int a_kind = ASRUtils::extract_kind_from_ttype_t(curr_type);
                switch (a_kind) {
                    case 1:
                        tmp = builder->CreateICmpNE(tmp, builder->getInt8(0));
                        break;
                    case 2:
                        tmp = builder->CreateICmpNE(tmp, builder->getInt16(0));
                        break;
                    case 4:
                        tmp = builder->CreateICmpNE(tmp, builder->getInt32(0));
                        break;
                    case 8:
                        tmp = builder->CreateICmpNE(tmp, builder->getInt64(0));
                        break;
                }
                break;
            }
            case (ASR::cast_kindType::RealToLogical) : {
                llvm::Value *zero;
                ASR::ttype_t* curr_type = extract_ttype_t_from_expr(x.m_arg);
                LCOMPILERS_ASSERT(curr_type != nullptr)
                int a_kind = ASRUtils::extract_kind_from_ttype_t(curr_type);
                if (a_kind == 4) {
                    zero = llvm::ConstantFP::get(context, llvm::APFloat((float)0.0));
                } else {
                    zero = llvm::ConstantFP::get(context, llvm::APFloat(0.0));
                }
                tmp = builder->CreateFCmpUNE(tmp, zero);
                break;
            }
            case (ASR::cast_kindType::StringToLogical) : {
                tmp = builder->CreateICmpNE(get_string_length(x.m_arg), builder->getInt32(0));
                break;
            }
            case (ASR::cast_kindType::StringToInteger) : {
                llvm::AllocaInst *parg = llvm_utils->CreateAlloca(*builder, character_type);
                builder->CreateStore(tmp, parg);
                tmp = lfortran_str_to_int(parg);
                break;
            }
            case (ASR::cast_kindType::ComplexToLogical) : {
                // !(c.real == 0.0 && c.imag == 0.0)
                llvm::Value *zero;
                ASR::ttype_t* curr_type = extract_ttype_t_from_expr(x.m_arg);
                LCOMPILERS_ASSERT(curr_type != nullptr)
                int a_kind = ASRUtils::extract_kind_from_ttype_t(curr_type);
                if (a_kind == 4) {
                    zero = llvm::ConstantFP::get(context, llvm::APFloat((float)0.0));
                } else {
                    zero = llvm::ConstantFP::get(context, llvm::APFloat(0.0));
                }
                llvm::Value *c_real = complex_re(tmp, tmp->getType());
                llvm::Value *real_check = builder->CreateFCmpUEQ(c_real, zero);
                llvm::Value *c_imag = complex_im(tmp, tmp->getType());
                llvm::Value *imag_check = builder->CreateFCmpUEQ(c_imag, zero);
                tmp = builder->CreateAnd(real_check, imag_check);
                tmp = builder->CreateNot(tmp);
                break;
            }
            case (ASR::cast_kindType::LogicalToInteger) : {
                int a_kind = ASRUtils::extract_kind_from_ttype_t(x.m_type);
                tmp = builder->CreateZExt(tmp, llvm_utils->getIntType(a_kind));
                break;
            }
            case (ASR::cast_kindType::RealToReal) : {
                int arg_kind = -1, dest_kind = -1;
                extract_kinds(x, arg_kind, dest_kind);
                if( arg_kind > 0 && dest_kind > 0 &&
                    arg_kind != dest_kind )
                {
                    if( arg_kind == 4 && dest_kind == 8 ) {
                        tmp = builder->CreateFPExt(tmp, llvm::Type::getDoubleTy(context));
                    } else if( arg_kind == 8 && dest_kind == 4 ) {
                        tmp = builder->CreateFPTrunc(tmp, llvm::Type::getFloatTy(context));
                    } else {
                        std::string msg = "Conversion from " + std::to_string(arg_kind) +
                                          " to " + std::to_string(dest_kind) + " not implemented yet.";
                        throw CodeGenError(msg);
                    }
                }
                break;
            }
            case (ASR::cast_kindType::IntegerToInteger) : {
                int arg_kind = -1, dest_kind = -1;
                extract_kinds(x, arg_kind, dest_kind);
                if( arg_kind > 0 && dest_kind > 0 &&
                    arg_kind != dest_kind )
                {
                    if (dest_kind > arg_kind) {
                        tmp = builder->CreateSExt(tmp, llvm_utils->getIntType(dest_kind));
                    } else {
                        tmp = builder->CreateTrunc(tmp, llvm_utils->getIntType(dest_kind));
                    }
                }
                break;
            }
            case (ASR::cast_kindType::UnsignedIntegerToUnsignedInteger) : {
                int arg_kind = -1, dest_kind = -1;
                extract_kinds(x, arg_kind, dest_kind);
                if( arg_kind > 0 && dest_kind > 0 &&
                    arg_kind != dest_kind )
                {
                    if (dest_kind > arg_kind) {
                        tmp = builder->CreateZExt(tmp, llvm_utils->getIntType(dest_kind));
                    } else {
                        tmp = builder->CreateTrunc(tmp, llvm_utils->getIntType(dest_kind));
                    }
                }
                break;
            }
            case (ASR::cast_kindType::IntegerToUnsignedInteger) : {
                int arg_kind = -1, dest_kind = -1;
                extract_kinds(x, arg_kind, dest_kind);
                LCOMPILERS_ASSERT(arg_kind != -1 && dest_kind != -1)
                if( arg_kind > 0 && dest_kind > 0 &&
                    arg_kind != dest_kind )
                {
                    if (dest_kind > arg_kind) {
                        tmp = builder->CreateSExt(tmp, llvm_utils->getIntType(dest_kind));
                    } else {
                        tmp = builder->CreateTrunc(tmp, llvm_utils->getIntType(dest_kind));
                    }
                }
                break;
            }
            case (ASR::cast_kindType::UnsignedIntegerToInteger) : {
                int arg_kind = -1, dest_kind = -1;
                extract_kinds(x, arg_kind, dest_kind);
                LCOMPILERS_ASSERT(arg_kind != -1 && dest_kind != -1)
                if( arg_kind > 0 && dest_kind > 0 &&
                    arg_kind != dest_kind )
                {
                    if (dest_kind > arg_kind) {
                        tmp = builder->CreateZExt(tmp, llvm_utils->getIntType(dest_kind));
                    } else {
                        tmp = builder->CreateTrunc(tmp, llvm_utils->getIntType(dest_kind));
                    }
                }
                break;
            }
            case (ASR::cast_kindType::CPtrToUnsignedInteger) : {
                tmp = builder->CreatePtrToInt(tmp, llvm_utils->getIntType(8, false));
                break;
            }
            case (ASR::cast_kindType::UnsignedIntegerToCPtr) : {
                tmp = builder->CreateIntToPtr(tmp, llvm::Type::getVoidTy(context)->getPointerTo());
                break;
            }
            case (ASR::cast_kindType::ComplexToComplex) : {
                llvm::Type *target_type;
                int arg_kind = -1, dest_kind = -1;
                extract_kinds(x, arg_kind, dest_kind);
                llvm::Value *re, *im;
                if( arg_kind > 0 && dest_kind > 0 &&
                    arg_kind != dest_kind )
                {
                    if( arg_kind == 4 && dest_kind == 8 ) {
                        target_type = complex_type_8;
                        re = complex_re(tmp, complex_type_4);
                        re = builder->CreateFPExt(re, llvm::Type::getDoubleTy(context));
                        im = complex_im(tmp, complex_type_4);
                        im = builder->CreateFPExt(im, llvm::Type::getDoubleTy(context));
                    } else if( arg_kind == 8 && dest_kind == 4 ) {
                        target_type = complex_type_4;
                        re = complex_re(tmp, complex_type_8);
                        re = builder->CreateFPTrunc(re, llvm::Type::getFloatTy(context));
                        im = complex_im(tmp, complex_type_8);
                        im = builder->CreateFPTrunc(im, llvm::Type::getFloatTy(context));
                    } else {
                        std::string msg = "Conversion from " + std::to_string(arg_kind) +
                                          " to " + std::to_string(dest_kind) + " not implemented yet.";
                        throw CodeGenError(msg);
                    }
                } else {
                    throw CodeGenError("Negative kinds are not supported.");
                }
                tmp = complex_from_floats(re, im, target_type);
                break;
            }
            case (ASR::cast_kindType::ComplexToReal) : {
                int arg_kind = -1, dest_kind = -1;
                extract_kinds(x, arg_kind, dest_kind);
                llvm::Value *re;
                if( arg_kind > 0 && dest_kind > 0)
                {
                    if( arg_kind == 4 && dest_kind == 4 ) {
                        // complex(4) -> real(4)
                        re = complex_re(tmp, complex_type_4);
                        tmp = re;
                    } else if( arg_kind == 4 && dest_kind == 8 ) {
                        // complex(4) -> real(8)
                        re = complex_re(tmp, complex_type_4);
                        tmp = builder->CreateFPExt(re, llvm::Type::getDoubleTy(context));
                    } else if( arg_kind == 8 && dest_kind == 4 ) {
                        // complex(8) -> real(4)
                        re = complex_re(tmp, complex_type_8);
                        tmp = builder->CreateFPTrunc(re, llvm::Type::getFloatTy(context));
                    } else if( arg_kind == 8 && dest_kind == 8 ) {
                        // complex(8) -> real(8)
                        re = complex_re(tmp, complex_type_8);
                        tmp = re;
                    } else {
                        std::string msg = "Conversion from " + std::to_string(arg_kind) +
                                          " to " + std::to_string(dest_kind) + " not implemented yet.";
                        throw CodeGenError(msg);
                    }
                } else {
                    throw CodeGenError("Negative kinds are not supported.");
                }
                break;
            }
            case (ASR::cast_kindType::ComplexToInteger) : {
                int arg_kind = -1, dest_kind = -1;
                extract_kinds(x, arg_kind, dest_kind);
                llvm::Value *re;
                if (arg_kind > 0 && dest_kind > 0)
                {
                    if (arg_kind == 4) {
                        // complex(4) -> real(8)
                        re = complex_re(tmp, complex_type_4);
                        tmp = re;
                    } else if (arg_kind == 8) {
                        // complex(8) -> real(8)
                        re = complex_re(tmp, complex_type_8);
                        tmp = re;
                    } else {
                        std::string msg = "Unsupported Complex type kind: " + std::to_string(arg_kind);
                        throw CodeGenError(msg);
                    }
                    llvm::Type *target_type;
                    target_type = llvm_utils->getIntType(dest_kind);
                    tmp = builder->CreateFPToSI(tmp, target_type);
                } else {
                    throw CodeGenError("Negative kinds are not supported.");
                }
                break;
             }
            case (ASR::cast_kindType::RealToString) : {
                llvm::Value *arg = tmp;
                ASR::ttype_t* arg_type = extract_ttype_t_from_expr(x.m_arg);
                LCOMPILERS_ASSERT(arg_type != nullptr)
                int arg_kind = ASRUtils::extract_kind_from_ttype_t(arg_type);
                tmp = lfortran_type_to_str(arg, llvm_utils->getFPType(arg_kind), "float", arg_kind);
                break;
            }
            case (ASR::cast_kindType::IntegerToString) : {
                llvm::Value *arg = tmp;
                ASR::ttype_t* arg_type = extract_ttype_t_from_expr(x.m_arg);
                LCOMPILERS_ASSERT(arg_type != nullptr)
                int arg_kind = ASRUtils::extract_kind_from_ttype_t(arg_type);
                tmp = lfortran_type_to_str(arg, llvm_utils->getIntType(arg_kind), "int", arg_kind);
                break;
            }
            case (ASR::cast_kindType::LogicalToString) : {
                llvm::Value *cmp = builder->CreateICmpEQ(tmp, builder->getInt1(0));
                llvm::Value *zero_str = builder->CreateGlobalStringPtr("False");
                llvm::Value *one_str = builder->CreateGlobalStringPtr("True");
                tmp = builder->CreateSelect(cmp, zero_str, one_str);
                break;
            }
            case (ASR::cast_kindType::ListToArray) : {
                if( !ASR::is_a<ASR::List_t>(*ASRUtils::expr_type(x.m_arg)) ) {
                    throw CodeGenError("The argument of ListToArray cast should "
                        "be a list/std::vector, found, " + ASRUtils::type_to_str_fortran(
                            ASRUtils::expr_type(x.m_arg)));
                }
                int64_t ptr_loads_copy = ptr_loads;
                ptr_loads = 0;
                this->visit_expr(*x.m_arg);
                ptr_loads = ptr_loads_copy;
                llvm::Type* list_llvm_type = llvm_utils->get_type_from_ttype_t_util(x.m_arg, ASRUtils::expr_type(x.m_arg), module.get());
                tmp = llvm_utils->CreateLoad(list_api->get_pointer_to_list_data_using_type(list_llvm_type, tmp));
                break;
            }
            case (ASR::cast_kindType::PointerToInteger): {
                llvm::Value *ptr = nullptr;
                if (ASR::is_a<ASR::Var_t>(*x.m_arg)) {
                    ptr = get_pointer_to_variable(x.m_arg);
                } else {
                    this->visit_expr(*x.m_arg);
                    llvm::Value *val = tmp;
                    llvm::AllocaInst *alloc = builder->CreateAlloca(val->getType(), nullptr);
                    builder->CreateStore(val, alloc);
                    ptr = alloc;
                }
                tmp = builder->CreatePtrToInt(ptr, llvm::Type::getInt64Ty(context));
                break;
            }
            default : throw CodeGenError("Cast kind not implemented");
        }
        ptr_loads = ptr_loads_copy;
    }

    llvm::Function* get_read_function(ASR::ttype_t *type) {
        type = ASRUtils::type_get_past_allocatable(
            ASRUtils::type_get_past_pointer(type));
        llvm::Function *fn = nullptr;
        switch (type->type) {
            case (ASR::ttypeType::Integer): {
                std::string runtime_func_name;
                llvm::Type *type_arg;
                int a_kind = ASRUtils::extract_kind_from_ttype_t(type);
                if ( a_kind == 2 ) {
                    runtime_func_name = "_lfortran_read_int16";
                    type_arg = llvm::Type::getInt16Ty(context);
                } else if (a_kind == 4) {
                    runtime_func_name = "_lfortran_read_int32";
                    type_arg = llvm::Type::getInt32Ty(context);
                } else if (a_kind == 8) {
                    runtime_func_name = "_lfortran_read_int64";
                    type_arg = llvm::Type::getInt64Ty(context);
                } else {
                    throw CodeGenError("Read Integer function not implemented "
                        "for integer kind: " + std::to_string(a_kind));
                }
                fn = module->getFunction(runtime_func_name);
                if (!fn) {
                    llvm::FunctionType *function_type = llvm::FunctionType::get(
                            llvm::Type::getVoidTy(context), {
                                type_arg->getPointerTo(),
                                llvm::Type::getInt32Ty(context)
                            }, false);
                    fn = llvm::Function::Create(function_type,
                            llvm::Function::ExternalLinkage, runtime_func_name, *module);
                }
                break;
            }
            case (ASR::ttypeType::String): {
                std::string runtime_func_name = "_lfortran_read_char";
                fn = module->getFunction(runtime_func_name);
                if (!fn) {
                    llvm::FunctionType *function_type = llvm::FunctionType::get(
                            llvm::Type::getVoidTy(context), {
                                character_type->getPointerTo(), // Str_data
                                llvm::Type::getInt64Ty(context), // Str_len
                                llvm::Type::getInt32Ty(context) // Unit_num
                            }, false);
                    fn = llvm::Function::Create(function_type,
                            llvm::Function::ExternalLinkage, runtime_func_name, *module);
                }
                break;
            }

            // adding case of boolean type input (TO DO:- boolean array)
            case (ASR::ttypeType::Logical):{
                std::string runtime_func_name;
                llvm::Type *type_arg;
                int a_kind = ASRUtils::extract_kind_from_ttype_t(type);

                if (a_kind == 4) {
                    runtime_func_name = "_lfortran_read_logical";
                    type_arg = llvm::Type::getInt1Ty(context); // LLVM boolean type (1 bit)
                } else {
                    throw CodeGenError("Read Logical function not implemented for kind: " + std::to_string(a_kind));
                }

                fn = module->getFunction(runtime_func_name);
                if (!fn) {
                    llvm::FunctionType *function_type = llvm::FunctionType::get(
                            llvm::Type::getVoidTy(context), {
                                type_arg->getPointerTo(),
                                llvm::Type::getInt32Ty(context)
                            }, false);
                    fn = llvm::Function::Create(function_type,
                            llvm::Function::ExternalLinkage, runtime_func_name, *module);
                }
                break;
            }
            case (ASR::ttypeType::Real): {
                std::string runtime_func_name;
                llvm::Type *type_arg;
                int a_kind = ASRUtils::extract_kind_from_ttype_t(type);
                if (a_kind == 4) {
                    runtime_func_name = "_lfortran_read_float";
                    type_arg = llvm::Type::getFloatTy(context);
                } else {
                    runtime_func_name = "_lfortran_read_double";
                    type_arg = llvm::Type::getDoubleTy(context);
                }
                fn = module->getFunction(runtime_func_name);
                if (!fn) {
                    llvm::FunctionType *function_type = llvm::FunctionType::get(
                            llvm::Type::getVoidTy(context), {
                                type_arg->getPointerTo(),
                                llvm::Type::getInt32Ty(context)
                            }, false);
                    fn = llvm::Function::Create(function_type,
                            llvm::Function::ExternalLinkage, runtime_func_name, *module);
                }
                break;
            }
            case (ASR::ttypeType::Array): {
                type = ASRUtils::type_get_past_array(type);
                int a_kind = ASRUtils::extract_kind_from_ttype_t(type);
                std::string runtime_func_name;
                llvm::Type *type_arg;
                if (ASR::is_a<ASR::Integer_t>(*type)) {
                    if (a_kind == 1) {
                        runtime_func_name = "_lfortran_read_array_int8";
                        type_arg = llvm::Type::getInt8Ty(context);
                    } else if ( a_kind == 2) {
                        runtime_func_name = "_lfortran_read_array_int16";
                        type_arg = llvm::Type::getInt16Ty(context);
                    } else if (a_kind == 4) {
                        runtime_func_name = "_lfortran_read_array_int32";
                        type_arg = llvm::Type::getInt32Ty(context);
                    } else if (a_kind == 8) {
                        runtime_func_name = "_lfortran_read_array_int64";
                        type_arg = llvm::Type::getInt64Ty(context);
                    } else {
                        throw CodeGenError("Integer arrays of kind 1 or 4 only supported for now. Found kind: "
                                            + std::to_string(a_kind));
                    }
                } else if (ASR::is_a<ASR::Real_t>(*type)) {
                    if (a_kind == 4) {
                        runtime_func_name = "_lfortran_read_array_float";
                        type_arg = llvm::Type::getFloatTy(context);
                    } else if (a_kind == 8) {
                        runtime_func_name = "_lfortran_read_array_double";
                        type_arg = llvm::Type::getDoubleTy(context);
                    } else {
                        throw CodeGenError("Real arrays of kind 4 or 8 only supported for now. Found kind: "
                                            + std::to_string(a_kind));
                    }
                } else if (ASR::is_a<ASR::Complex_t>(*type)) {
                    if ( a_kind == 4 ) {
                        runtime_func_name = "_lfortran_read_array_complex_float";
                        type_arg = complex_type_4;
                    } else if ( a_kind == 8 ) {
                        runtime_func_name = "_lfortran_read_array_complex_double";
                        type_arg = complex_type_8;
                    } else {
                        throw CodeGenError("Complex arrays of kind 4 or 8 only supported for now. Found kind: "
                                            + std::to_string(a_kind));
                    }
                } else if (ASR::is_a<ASR::String_t>(*type)) {
                    runtime_func_name = "_lfortran_read_array_char";
                    type_arg = llvm::Type::getInt8Ty(context);
                } else {
                    throw CodeGenError("Type not supported.");
                }
                fn = module->getFunction(runtime_func_name);
                if (!fn) {
                    std::vector<llvm::Type*> types {
                        type_arg->getPointerTo(), 
                        llvm::Type::getInt32Ty(context),
                        llvm::Type::getInt32Ty(context)};
                    if(runtime_func_name == "_lfortran_read_array_char"){
                        types.insert(types.begin()+1, llvm::Type::getInt64Ty(context));
                    }
                    llvm::FunctionType *function_type = llvm::FunctionType::get(
                        llvm::Type::getVoidTy(context), types, false);
                    fn = llvm::Function::Create(function_type,
                            llvm::Function::ExternalLinkage, runtime_func_name, *module);
                }
                break;
            }
            default: {
                std::string s_type = ASRUtils::type_to_str_fortran(type);
                throw CodeGenError("Read function not implemented for: " + s_type);
            }
        }
        return fn;
    }

    void visit_FileRead(const ASR::FileRead_t &x) {
        if( x.m_overloaded ) {
            this->visit_stmt(*x.m_overloaded);
            return ;
        }

        llvm::Value *unit_val, *iostat, *read_size;
        llvm::Value *advance, *advance_length;
        bool is_string = false;
        if (x.m_unit == nullptr) {
            // Read from stdin
            unit_val = llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(context), llvm::APInt(32, -1, true));
        } else {
            is_string = ASRUtils::is_character(*expr_type(x.m_unit));
            this->visit_expr_load_wrapper(x.m_unit, (is_string ? 0 : 1), true);
            unit_val = tmp;
            if(ASRUtils::is_integer(*ASRUtils::expr_type(x.m_unit))){
                // Convert the unit to 32 bit integer (We only support unit number up to 1000).
                unit_val = llvm_utils->convert_kind(tmp, llvm::Type::getInt32Ty(context));
            }
        }

        if (x.m_iostat) {
            int ptr_copy = ptr_loads;
            ptr_loads = 0;
            this->visit_expr_wrapper(x.m_iostat, false);
            ptr_loads = ptr_copy;
            iostat = tmp;
        } else {
            iostat = llvm_utils->CreateAlloca(*builder,
                        llvm::Type::getInt32Ty(context));
        }

        if (x.m_advance) {
            this->visit_expr_wrapper(x.m_advance, true);
            LCOMPILERS_ASSERT(ASRUtils::is_character(*expr_type(x.m_advance)))
            std::tie(advance, advance_length) = llvm_utils->get_string_length_data(
                                                    ASRUtils::get_string_type(x.m_advance),
                                                    tmp);
        } else {
            std::string yes("yes");
            advance = builder->CreateGlobalStringPtr(yes);
            advance_length = llvm::ConstantInt::get(context, llvm::APInt(64, yes.size()));
        }

        if (x.m_size) {
            int ptr_copy = ptr_loads;
            ptr_loads = 0;
            this->visit_expr_wrapper(x.m_size, false);
            ptr_loads = ptr_copy;
            read_size = tmp;
        } else {
            read_size = llvm_utils->CreateAlloca(*builder,
                        llvm::Type::getInt32Ty(context));
        }

        if (x.m_fmt) {
            std::vector<llvm::Value*> args;
            args.push_back(unit_val);
            args.push_back(iostat);
            args.push_back(read_size);
            args.push_back(advance);
            args.push_back(advance_length);
            this->visit_expr_wrapper(x.m_fmt, true);
            args.push_back(llvm_utils->get_string_data(ASRUtils::get_string_type(expr_type(x.m_fmt)), tmp));
            args.push_back(llvm::ConstantInt::get(context, llvm::APInt(32, x.n_values * 2 /*(str_data, str_len)*/)));
            for (size_t i=0; i<x.n_values; i++) {
                this->visit_expr_load_wrapper(x.m_values[i], 0);
                llvm::Value* str_data, *str_len;
                std::tie(str_data, str_len) = 
                    llvm_utils->get_string_length_data(
                        ASRUtils::get_string_type(expr_type(x.m_values[i])),
                        tmp);
                args.push_back(str_data);
                args.push_back(str_len);
            }
            std::string runtime_func_name = "_lfortran_formatted_read";
            llvm::Function *fn = module->getFunction(runtime_func_name);
            if (!fn) {
                llvm::FunctionType *function_type = llvm::FunctionType::get(
                        llvm::Type::getVoidTy(context), {
                            llvm::Type::getInt32Ty(context),
                            llvm::Type::getInt32Ty(context)->getPointerTo(),
                            llvm::Type::getInt32Ty(context)->getPointerTo(),
                            character_type, // advance
                            llvm::Type::getInt64Ty(context), // advance_length
                            character_type,
                            llvm::Type::getInt32Ty(context)
                        }, true);
                fn = llvm::Function::Create(function_type,
                        llvm::Function::ExternalLinkage, runtime_func_name, *module);
            }
            builder->CreateCall(fn, args);
        } else {
            llvm::Value* var_to_read_into = nullptr; // Var expression that we'll read into.
            for (size_t i=0; i<x.n_values; i++) {
                int ptr_copy = ptr_loads;
                ptr_loads = 0;
                this->visit_expr(*x.m_values[i]);
                var_to_read_into = tmp; tmp =nullptr;
                ptr_loads = ptr_copy;
                ASR::ttype_t* type = ASRUtils::expr_type(x.m_values[i]);
                llvm::Function *fn;
                if (ASR::is_a<ASR::Var_t>(*x.m_values[i]) &&
                    ASR::is_a<ASR::ExternalSymbol_t>(*ASR::down_cast<ASR::Var_t>(x.m_values[i])->m_v)) {
                    ASR::Variable_t *asr_target = EXPR2VAR(x.m_values[i]);
                    uint32_t h = get_hash((ASR::asr_t*)asr_target);
                    var_to_read_into = llvm_symtab[h];
                }
                if (is_string) {
                    // TODO: Support multiple arguments and fmt
                    std::string runtime_func_name = "_lfortran_string_read_" +
                                            ASRUtils::type_to_str_python(ASRUtils::extract_type(type));
                    if (ASRUtils::is_array(type)) {
                        runtime_func_name += "_array";
                    }
                    llvm::Function* fn = module->getFunction(runtime_func_name);
                    if (!fn) {
                        llvm::FunctionType *function_type;
                        if(ASRUtils::is_string_only(type)){
                            function_type = llvm::FunctionType::get(
                                llvm::Type::getVoidTy(context),
                                {
                                    character_type /*src_data*/,
                                    llvm::Type::getInt64Ty(context) /*src_length*/,
                                    character_type /*dest_data*/,
                                    llvm::Type::getInt64Ty(context) /*dest_length*/
                                },
                                false);
                        } else {
                            function_type = llvm::FunctionType::get(
                                llvm::Type::getVoidTy(context),
                                {   character_type /*src_data*/,
                                    llvm::Type::getInt64Ty(context)/*src_length*/,
                                    character_type,
                                    llvm_utils->get_type_from_ttype_t_util(
                                        x.m_values[i], type, module.get()
                                    )->getPointerTo()
                                },
                                false);
                        }
                        fn = llvm::Function::Create(function_type,
                                llvm::Function::ExternalLinkage, runtime_func_name, *module);
                    }
                    llvm::Value *fmt = nullptr;
                    if (ASR::is_a<ASR::Integer_t>(*ASRUtils::type_get_past_array(
                            ASRUtils::type_get_past_allocatable_pointer(type)))) {
                        ASR::Integer_t* int_type = ASR::down_cast<ASR::Integer_t>(ASRUtils::type_get_past_array(
                                ASRUtils::type_get_past_allocatable_pointer(type)));
                        fmt = int_type->m_kind == 4 ? builder->CreateGlobalStringPtr("%d")
                                                    : builder->CreateGlobalStringPtr("%ld");
                    } else if (ASR::is_a<ASR::Real_t>(*ASRUtils::type_get_past_array(
                                   ASRUtils::type_get_past_allocatable_pointer(type)))) {
                        ASR::Real_t* real_type = ASR::down_cast<ASR::Real_t>(ASRUtils::type_get_past_array(
                                ASRUtils::type_get_past_allocatable_pointer(type)));
                        fmt = real_type->m_kind == 4 ? builder->CreateGlobalStringPtr("%f")
                                                     : builder->CreateGlobalStringPtr("%lf");
                    } else if (ASR::is_a<ASR::String_t>(*ASRUtils::type_get_past_array(
                                   ASRUtils::type_get_past_allocatable_pointer(type)))) {
                        fmt = builder->CreateGlobalStringPtr("%s");
                    } else if (ASR::is_a<ASR::Logical_t>(*ASRUtils::type_get_past_array(
                                   ASRUtils::type_get_past_allocatable_pointer(type)))) {
                        fmt = builder->CreateGlobalStringPtr("%d");
                    }
                    llvm::Value *src_data, *src_len;
                    std::tie(src_data, src_len) = llvm_utils->get_string_length_data(
                        ASRUtils::get_string_type(x.m_unit), unit_val);

                    if(ASRUtils::is_string_only(type)){
                        llvm::Value* dest_data, *dest_len;
                        std::tie(dest_data, dest_len) = llvm_utils->get_string_length_data(
                            ASRUtils::get_string_type(x.m_values[i]), var_to_read_into);
                        builder->CreateCall(fn, { src_data, src_len, dest_data, dest_len });
                    } else {
                        builder->CreateCall(fn, { src_data, src_len, fmt, var_to_read_into });
                    }
                    return;
                } else {
                    fn = get_read_function(type);
                }
                if (ASRUtils::is_array(type)) {
                    llvm::Type *el_type = llvm_utils->get_type_from_ttype_t_util(x.m_values[i], ASRUtils::extract_type(type), module.get());
                    if (ASR::is_a<ASR::Allocatable_t>(*type)
                        || ASR::is_a<ASR::Pointer_t>(*type)) {
                        var_to_read_into = llvm_utils->CreateLoad(var_to_read_into);
                    }
                    llvm::Value* original_array_representation = var_to_read_into; // Loaded (if necessary)
#if LLVM_VERSION_MAJOR > 16
                    ptr_type[var_to_read_into] = llvm_utils->get_type_from_ttype_t_util(x.m_values[i],
                        ASRUtils::type_get_past_allocatable_pointer(type),
                        module.get());
#endif
                    ASR::Array_t *arr_tp = ASR::down_cast<ASR::Array_t>(
                        ASRUtils::type_get_past_allocatable_pointer(type));
                    if (arr_tp->m_physical_type != ASR::array_physical_typeType::PointerToDataArray) {
                        var_to_read_into = arr_descr->get_pointer_to_data(var_to_read_into);
                    }
                    if (ASR::is_a<ASR::Allocatable_t>(*type)
                        || ASR::is_a<ASR::Pointer_t>(*type)) {
                        var_to_read_into = llvm_utils->CreateLoad2(el_type->getPointerTo(), var_to_read_into);
                    }
                    llvm::Value *arr = var_to_read_into;
                    ASR::ttype_t *type32 = ASRUtils::TYPE(ASR::make_Integer_t(al, x.base.base.loc, 4));
                    ASR::ArraySize_t* array_size = ASR::down_cast2<ASR::ArraySize_t>(ASR::make_ArraySize_t(al, x.base.base.loc,
                        x.m_values[i], nullptr, type32, nullptr));
                    visit_ArraySize(*array_size);
                    if(ASRUtils::is_array_of_strings(type)){
                        builder->CreateCall(fn, {
                            llvm_utils->get_stringArray_data(type, original_array_representation),
                            llvm_utils->get_stringArray_length(type, original_array_representation),
                            tmp, unit_val});
                        tmp = nullptr;
                    } else {
                        builder->CreateCall(fn, {arr, tmp, unit_val}); tmp = nullptr;
                    }
                } else {
                    if(ASRUtils::is_string_only(type)){
                        llvm::Value* str_data, *str_len;
                        std::tie(str_data, str_len) = llvm_utils->get_string_length_data(ASRUtils::get_string_type(type), var_to_read_into, true);
                        builder->CreateCall(fn, {str_data, str_len, unit_val});
                    } else {
                        builder->CreateCall(fn, {var_to_read_into, unit_val});
                    }
                }
            }

            // In Fortran, read(u, *) is used to read the entire line. The
            // next read(u, *) function is intended to read the next entire
            // line. Let's take an example: `read(u, *) n`, where n is an
            // integer. The first occurance of the integer value will be
            // read, and anything after that will be skipped.
            // Here, we can use `_lfortran_empty_read` function to move to the
            // pointer to the next line.
            std::string runtime_func_name = "_lfortran_empty_read";
            llvm::Function *fn = module->getFunction(runtime_func_name);
            if (!fn) {
                llvm::FunctionType *function_type = llvm::FunctionType::get(
                        llvm::Type::getVoidTy(context), {
                            llvm::Type::getInt32Ty(context),
                            llvm::Type::getInt32Ty(context)->getPointerTo()
                        }, false);
                fn = llvm::Function::Create(function_type,
                        llvm::Function::ExternalLinkage, runtime_func_name,
                            *module);
            }
            builder->CreateCall(fn, {unit_val, iostat});
        }
    }

    void visit_FileOpen(const ASR::FileOpen_t &x) {
        llvm::Value *unit_val = nullptr; 
        llvm::Value* f_name_data{}, *f_name_len{};
        llvm::Value *status_data{}, *status_len{};
        llvm::Value *form_data{},   *form_len{};
        llvm::Value *access_data{}, *access_len{};
        llvm::Value *iomsg_data{},  *iomsg_len{};
        llvm::Value *iostat{};
        llvm::Value *action{}, *action_len{};
        
        this->visit_expr_wrapper(x.m_newunit, true);
        unit_val = llvm_utils->convert_kind(tmp, llvm::Type::getInt32Ty(context));
        int ptr_copy = ptr_loads;
        if (x.m_filename) {
            std::tie(f_name_data, f_name_len) = get_string_data_and_length(x.m_filename);
        } else {
            f_name_data = llvm::Constant::getNullValue(character_type);
            f_name_len = llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), 0);
        }
        if (x.m_status) {
            std::tie(status_data, status_len) = get_string_data_and_length(x.m_status);
        } else {
            status_data = llvm::Constant::getNullValue(character_type);
            status_len = llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), 0);
        }
        if (x.m_form) {
            std::tie(form_data, form_len) = get_string_data_and_length(x.m_form);
        } else {
            form_data = llvm::Constant::getNullValue(character_type);
            form_len = llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), 0);
        }
        if (x.m_access) {
            std::tie(access_data, access_len) = get_string_data_and_length(x.m_access);
        } else {
            access_data = llvm::Constant::getNullValue(character_type);
            access_len = llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), 0);
        }
        if (x.m_iostat) {
            int ptr_copy = ptr_loads;
            ptr_loads = 0;
            this->visit_expr_wrapper(x.m_iostat, false);
            ptr_loads = ptr_copy;
            iostat = tmp;
        } else {
            iostat = llvm::ConstantInt::getNullValue(llvm::Type::getInt32Ty(context)->getPointerTo());
        }
        if (x.m_iomsg) {
            std::tie(iomsg_data, iomsg_len) = get_string_data_and_length(x.m_iomsg);
        } else {
            iomsg_data = llvm::Constant::getNullValue(character_type);
            iomsg_len = llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), 0);
        } if (x.m_action) {
            std::tie(action, action_len) = get_string_data_and_length(x.m_action);
        } else {
            action = llvm::Constant::getNullValue(character_type);
            action_len = llvm::ConstantInt::get(context, llvm::APInt(64, 0));
        }
        ptr_loads = ptr_copy;
        std::string runtime_func_name = "_lfortran_open";
        llvm::Function *fn = module->getFunction(runtime_func_name);
        if (!fn) {
            llvm::Type *i64 = llvm::Type::getInt64Ty(context);
            llvm::FunctionType *function_type = llvm::FunctionType::get(
                    i64, 
                    {llvm::Type::getInt32Ty(context),
                        character_type, i64, //f_name, f_name_length
                        character_type, i64, //status, status_len 
                        character_type, i64, //form, form_length
                        character_type, i64, //access, access_len
                        character_type, i64, //iomsg, iomsg_len
                        llvm::Type::getInt32Ty(context)->getPointerTo(),
                        character_type, i64 // action, action_len
                    }, false);
            fn = llvm::Function::Create(function_type,
                    llvm::Function::ExternalLinkage, runtime_func_name, *module);
        }
        tmp = builder->CreateCall(fn, {unit_val,
            f_name_data,f_name_len, 
            status_data, status_len,
            form_data, form_len,
            access_data, access_len,
            iomsg_data, iomsg_len,
            iostat,
            action, action_len});
    }

    void visit_FileInquire(const ASR::FileInquire_t &x) {
        llvm::Value *exist_val {};
        llvm::Value* f_name_data {}, *f_name_len{};
        llvm::Value *unit {}, *opened_val {}, *size_val {}, *pos_val {};
        llvm::Value *write{}, *write_len{};
        llvm::Value *read{}, *read_len{};
        llvm::Value *readwrite{}, *readwrite_len{};

        if (x.m_file) {
            std::tie(f_name_data, f_name_len) = get_string_data_and_length(x.m_file);
        } else {
            f_name_data = llvm::Constant::getNullValue(character_type);
            f_name_len = llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), 0);
        }
        if (x.m_exist) {
            int ptr_loads_copy = ptr_loads;
            ptr_loads = 0;
            this->visit_expr_wrapper(x.m_exist, true);
            exist_val = tmp;
            ptr_loads = ptr_loads_copy;
        } else {
            exist_val = llvm_utils->CreateAlloca(*builder,
                            llvm::Type::getInt1Ty(context));
        }

        if (x.m_unit) {
            this->visit_expr_wrapper(x.m_unit, true);
            unit = tmp;
        } else {
            unit = llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(context), llvm::APInt(32, -1, true));
        }
        if (x.m_opened) {
            int ptr_loads_copy = ptr_loads;
            ptr_loads = 0;
            this->visit_expr_wrapper(x.m_opened, true);
            opened_val = tmp;
            ptr_loads = ptr_loads_copy;
        } else {
            opened_val = llvm_utils->CreateAlloca(*builder,
                            llvm::Type::getInt1Ty(context));
        }

        if (x.m_size) {
            int ptr_loads_copy = ptr_loads;
            ptr_loads = 0;
            this->visit_expr_wrapper(x.m_size, true);
            size_val = tmp;
            ptr_loads = ptr_loads_copy;
        } else {
            size_val = llvm_utils->CreateAlloca(*builder,
                            llvm::Type::getInt32Ty(context));
        }

        if (x.m_pos) {
            int ptr_loads_copy = ptr_loads;
            ptr_loads = 0;
            this->visit_expr_wrapper(x.m_pos, true);
            pos_val = tmp;
            ptr_loads = ptr_loads_copy;
        } else {
            pos_val = llvm_utils->CreateAlloca(*builder,
                            llvm::Type::getInt32Ty(context));
        }

        if (x.m_write) {
            this->visit_expr_load_wrapper(x.m_write, 0);
            std::tie(write, write_len) = 
                llvm_utils->get_string_length_data(
                    ASRUtils::get_string_type(x.m_write),
                    tmp);
        } else {
            write = llvm::Constant::getNullValue(character_type);
            write_len = llvm::ConstantInt::get(context, llvm::APInt(64, 0));
        }

        if (x.m_read) {
            this->visit_expr_load_wrapper(x.m_read, 0);
            std::tie(read, read_len) = 
                llvm_utils->get_string_length_data(
                    ASRUtils::get_string_type(x.m_read),
                    tmp);
        } else {
            read = llvm::Constant::getNullValue(character_type);
            read_len = llvm::ConstantInt::get(context, llvm::APInt(64, 0));
        }

        if (x.m_readwrite) {
            this->visit_expr_load_wrapper(x.m_readwrite, 0);
            std::tie(readwrite, readwrite_len) = 
                llvm_utils->get_string_length_data(
                    ASRUtils::get_string_type(x.m_readwrite),
                    tmp);
        } else {
            readwrite = llvm::Constant::getNullValue(character_type);
            readwrite_len = llvm::ConstantInt::get(context, llvm::APInt(64, 0));
        }

        std::string runtime_func_name = "_lfortran_inquire";
        llvm::Function *fn = module->getFunction(runtime_func_name);
        if (!fn) {
            llvm::FunctionType *function_type = llvm::FunctionType::get(
                    llvm::Type::getVoidTy(context), {
                        character_type, llvm::Type::getInt64Ty(context),
                        llvm::Type::getInt1Ty(context)->getPointerTo(),
                        llvm::Type::getInt32Ty(context),
                        llvm::Type::getInt1Ty(context)->getPointerTo(),
                        llvm::Type::getInt32Ty(context)->getPointerTo(),
                        llvm::Type::getInt32Ty(context)->getPointerTo(),
                        character_type, llvm::Type::getInt64Ty(context), // write_data, write_len
                        character_type, llvm::Type::getInt64Ty(context), // read_data, read_len
                        character_type, llvm::Type::getInt64Ty(context) // readwrite_data, readwrite_len
                    }, false);
            fn = llvm::Function::Create(function_type,
                    llvm::Function::ExternalLinkage, runtime_func_name, *module);
        }
        tmp = builder->CreateCall(fn, {f_name_data, f_name_len,
            exist_val, unit,
            opened_val, size_val,
            pos_val,
            write, write_len,
            read, read_len,
            readwrite, readwrite_len});
    }

    void visit_Flush(const ASR::Flush_t& x) {
        std::string runtime_func_name = "_lfortran_flush";
        llvm::Function *fn = module->getFunction(runtime_func_name);
        if (!fn) {
            llvm::FunctionType *function_type = llvm::FunctionType::get(
                    llvm::Type::getVoidTy(context), {
                        llvm::Type::getInt32Ty(context)
                    }, false);
            fn = llvm::Function::Create(function_type,
                    llvm::Function::ExternalLinkage, runtime_func_name, *module);
        }
        llvm::Value *unit_val = nullptr;
        this->visit_expr_wrapper(x.m_unit, true);
        unit_val = tmp;
        builder->CreateCall(fn, {unit_val});
    }

    void visit_FileRewind(const ASR::FileRewind_t &x) {
        std::string runtime_func_name = "_lfortran_rewind";
        llvm::Function *fn = module->getFunction(runtime_func_name);
        if (!fn) {
            llvm::FunctionType *function_type = llvm::FunctionType::get(
                    llvm::Type::getVoidTy(context), {
                        llvm::Type::getInt32Ty(context)
                    }, false);
            fn = llvm::Function::Create(function_type,
                    llvm::Function::ExternalLinkage, runtime_func_name, *module);
        }
        this->visit_expr_wrapper(x.m_unit, true);
        builder->CreateCall(fn, {tmp});
    }

    void visit_FileBackspace(const ASR::FileBackspace_t &x) {
        std::string runtime_func_name = "_lfortran_backspace";
        llvm::Function *fn = module->getFunction(runtime_func_name);
        if (!fn) {
            llvm::FunctionType *function_type = llvm::FunctionType::get(
                    llvm::Type::getVoidTy(context), {
                        llvm::Type::getInt32Ty(context)
                    }, false);
            fn = llvm::Function::Create(function_type,
                    llvm::Function::ExternalLinkage, runtime_func_name, *module);
        }
        this->visit_expr_wrapper(x.m_unit, true);
        builder->CreateCall(fn, {tmp});
    }

    void visit_FileClose(const ASR::FileClose_t &x) {
        llvm::Value *unit_val;
        llvm::Value *status, *status_len;
        this->visit_expr_wrapper(x.m_unit, true);
        unit_val = llvm_utils->convert_kind(tmp, llvm::Type::getInt32Ty(context));
        if (x.m_status) {
            this->visit_expr_load_wrapper(x.m_status, 0);
            status = tmp;
            std::tie(status, status_len) = llvm_utils->get_string_length_data(
                                            ASRUtils::get_string_type(x.m_status),
                                            tmp);
        } else {
            status = llvm::Constant::getNullValue(character_type);
            status_len = llvm::ConstantInt::get(context, llvm::APInt(64, 0));
        }
        std::string runtime_func_name = "_lfortran_close";
        llvm::Function *fn = module->getFunction(runtime_func_name);
        if (!fn) {
            llvm::FunctionType *function_type = llvm::FunctionType::get(
                    llvm::Type::getVoidTy(context), {
                        llvm::Type::getInt32Ty(context),
                        character_type, llvm::Type::getInt64Ty(context),
                    }, false);
            fn = llvm::Function::Create(function_type,
                    llvm::Function::ExternalLinkage, runtime_func_name, *module);
        }
        tmp = builder->CreateCall(fn, {unit_val, status, status_len});
    }

    void visit_Print(const ASR::Print_t &x) {
        handle_print(x.m_text, nullptr);
    }

    void visit_FileWrite(const ASR::FileWrite_t &x) {
        if( x.m_overloaded ) {
            this->visit_stmt(*x.m_overloaded);
            return ;
        }

        if (x.m_unit == nullptr) {
            if(x.n_values  == 0){ // TODO : We should remove any function that creates a `FileWrite` with no args
                llvm::Value *fmt_ptr = builder->CreateGlobalStringPtr("%s");
                llvm::Value* end{};
                if (x.m_end) { end = get_string_data(x.m_end); }
                printf(context, *module, *builder, {fmt_ptr, end});
            } else if (x.n_values == 1){
                handle_print(x.m_values[0], x.m_end);
            } else {
                throw CodeGenError("File write should have single argument of type character)", x.base.base.loc);
            }
            return;
        }
        std::vector<llvm::Value *> args;
        std::vector<llvm::Type *> args_type;
        std::vector<std::string> fmt;
        llvm::Value *sep_data {}, *sep_len {};
        llvm::Value *end_data {}, *end_len {};
        llvm::Value *unit = nullptr;
        llvm::Value *iostat = nullptr;
        std::string runtime_func_name;
        bool is_string = ASRUtils::is_character(*expr_type(x.m_unit));

        int ptr_loads_copy = ptr_loads;
        if ( is_string ) {
            ptr_loads = 0;
            runtime_func_name = "_lfortran_string_write";
            args_type.push_back(character_type->getPointerTo());// str_holder
            args_type.push_back(llvm::Type::getInt8Ty(context)); // is_allocatable
            args_type.push_back(llvm::Type::getInt8Ty(context)); // is_deferred
            args_type.push_back(llvm::Type::getInt64Ty(context)->getPointerTo()); //len
            args_type.push_back(llvm::Type::getInt32Ty(context)->getPointerTo()); //iostat
            args_type.push_back(llvm::Type::getInt8Ty(context)->getPointerTo()); //format_data
            args_type.push_back(llvm::Type::getInt64Ty(context)); // format_len
        } else if ( ASRUtils::is_integer(*expr_type(x.m_unit)) ) {
            ptr_loads = 1;
            runtime_func_name = "_lfortran_file_write";
            args_type.push_back(llvm::Type::getInt32Ty(context)); //unit_num
            args_type.push_back(llvm::Type::getInt32Ty(context)->getPointerTo()); //iostat
            args_type.push_back(llvm::Type::getInt8Ty(context)->getPointerTo()); //format_data
            args_type.push_back(llvm::Type::getInt64Ty(context));//format_len

        } else {
            throw CodeGenError("Unsupported type for `unit` in write(..)");
        }
        this->visit_expr_wrapper(x.m_unit);
        ptr_loads = ptr_loads_copy;

        llvm::Value* string_len;
        if(!is_string){
            unit = tmp;
            if (unit->getType()->isPointerTy()) {
                unit = llvm_utils->CreateLoad2(llvm::Type::getInt32Ty(context), unit);
            }
        } else { // String Write
            std::tie(unit, string_len) = llvm_utils->get_string_length_data(
                ASRUtils::get_string_type(x.m_unit), tmp, true, true);
        }
        ptr_loads = ptr_loads_copy;

        if (x.m_iostat) {
            int ptr_copy = ptr_loads;
            ptr_loads = 0;
            this->visit_expr_wrapper(x.m_iostat, false);
            ptr_loads = ptr_copy;
            iostat = tmp;
        } else {
            iostat = llvm_utils->CreateAlloca(*builder,
                        llvm::Type::getInt32Ty(context)->getPointerTo());
            builder->CreateStore(llvm::ConstantInt::getNullValue(
                llvm::Type::getInt32Ty(context)->getPointerTo()), iostat);
            iostat = llvm_utils->CreateLoad2(llvm::Type::getInt32Ty(context)->getPointerTo(), iostat);
        }

        if (x.m_separator) {
            std::tie(sep_data, sep_len) = get_string_data_and_length(x.m_separator);
        } else {
            sep_data = builder->CreateGlobalStringPtr(" ");
            sep_len = llvm::ConstantInt::get(context, llvm::APInt(64, 1));
        }
        if (x.m_end) {
            std::tie(end_data, end_len) = get_string_data_and_length(x.m_end);   
        } else {
            end_data = builder->CreateGlobalStringPtr("\n");
            end_len = llvm::ConstantInt::get(context, llvm::APInt(64, 1));
        }
        size_t n_values = x.n_values; ASR::expr_t **m_values = x.m_values;
        for (size_t i=0; i<n_values; i++) {
            if (i != 0 && !is_string && x.m_is_formatted) {
                fmt.push_back("%s");
                args.push_back(sep_data);
                args.push_back(sep_len);
            }
            if (!x.m_is_formatted) {
                int kind = ASRUtils::extract_kind_from_ttype_t(ASRUtils::extract_type(ASRUtils::expr_type(m_values[i])));
                llvm::Value* kind_val = llvm::ConstantInt::get(context, llvm::APInt(32, kind, true));
                ASR::ttype_t *type32 = ASRUtils::TYPE(ASR::make_Integer_t(al, x.base.base.loc, 4));
                if (ASRUtils::is_array(ASRUtils::expr_type(m_values[i]))) {
                    ASR::ArraySize_t* array_size = ASR::down_cast2<ASR::ArraySize_t>(ASR::make_ArraySize_t(al, m_values[i]->base.loc,
                        m_values[i], nullptr, type32, nullptr));
                    visit_ArraySize(*array_size);
                    args.push_back(builder->CreateMul(kind_val, tmp));
                } else if (ASRUtils::is_character(*ASRUtils::expr_type(m_values[i]))) {
                    ASR::StringLen_t * strlen = ASR::down_cast2<ASR::StringLen_t>(ASR::make_StringLen_t(al,
                        m_values[i]->base.loc, m_values[i], type32, nullptr));
                    visit_StringLen(*strlen);
                    args.push_back(builder->CreateMul(kind_val, tmp));
                } else {
                    args.push_back(kind_val);
                    this->visit_expr_wrapper(m_values[i], true);
                    llvm::Type* type = llvm_utils->get_type_from_ttype_t_util(m_values[i], ASRUtils::expr_type(m_values[i]), module.get());
                    llvm::Value* llvm_arg = llvm_utils->CreateAlloca(*builder, type);
                    builder->CreateStore(tmp, llvm_arg);
                    args.push_back(llvm_arg);
                    continue;
                    
                }
            }
            compute_fmt_specifier_and_arg(fmt, args, m_values[i],
                x.base.base.loc);
        }
        if (!x.m_is_formatted) {  // give -1 argument for end of arguments
            llvm::Value* minus_one = llvm::ConstantInt::get(context, llvm::APInt(32, -1, true));
            args.push_back(minus_one);
        } else if (!is_string) {
            fmt.push_back("%s");
            args.push_back(end_data);
            args.push_back(end_len);
        }
        std::string fmt_str;
        for (size_t i=0; i<fmt.size(); i++) {
            fmt_str += fmt[i];
        }
        llvm::Value *fmt_data = builder->CreateGlobalStringPtr(fmt_str);
        llvm::Value *fmt_len = llvm::ConstantInt::get(context, llvm::APInt(64, fmt_str.length()));
        

        std::vector<llvm::Value *> printf_args;
        printf_args.push_back(unit);
        if(is_string){
            llvm::Value* is_allocatable = llvm::ConstantInt::get(context, llvm::APInt(8,
                ASRUtils::is_allocatable(ASRUtils::expr_type(x.m_unit))));
            llvm::Value* is_deferred = llvm::ConstantInt::get(context, llvm::APInt(8,
                ASRUtils::is_deferredLength_string(ASRUtils::expr_type(x.m_unit))));
            printf_args.push_back(is_allocatable);
            printf_args.push_back(is_deferred);
            printf_args.push_back(string_len);
        }
        printf_args.push_back(iostat);
        printf_args.push_back(fmt_data);
        printf_args.push_back(fmt_len);
        printf_args.insert(printf_args.end(), args.begin(), args.end());
        llvm::Function *fn = module->getFunction(runtime_func_name);
        if (!fn) {

            llvm::FunctionType *function_type = llvm::FunctionType::get(
                    llvm::Type::getVoidTy(context), args_type, true);
            fn = llvm::Function::Create(function_type,
                    llvm::Function::ExternalLinkage, runtime_func_name, *module);
        }
        tmp = builder->CreateCall(fn, printf_args);
    }

    std::string serialize_structType_symbols(ASR::symbol_t* sym){
        std::string res {};
        ASR::Struct_t* StructSymbol = ASR::down_cast<ASR::Struct_t>(sym);
        for(size_t i=0; i < StructSymbol->n_members; i++){
            ASR::symbol_t* StructMember = StructSymbol->m_symtab->
                                            get_symbol(StructSymbol->m_members[i]);
            res += SerializeType(ASRUtils::get_expr_from_sym(al, StructMember), ASRUtils::symbol_type(StructMember), true);
            if(i < StructSymbol->n_members-1){
                res += ",";
            }
        }
        return res;
    }

    /*
        [ ] --> Array
        ( ) --> Struct
        I   --> Integer
        R   --> Real
        S-PhysicalType-len? --> Character [S-DESC-10, S-CCHAR-1, S-DESC]
        L   --> Logical
        { } --> Struct for COMPLEX
        The serialization corresponds to how these arguments are represented in LLVM backend;
        So, Complex type results in `{R, R}` as it's a struct of floating types in the backend.
    */

    // Serialize `type` using symbols above.
    std::string SerializeType(ASR::expr_t* expr, ASR::ttype_t* type, bool in_struct) {
        std::string res {};
        type = ASRUtils::type_get_past_allocatable(
                ASRUtils::type_get_past_pointer(type));
        if (ASRUtils::is_unlimited_polymorphic_type(expr)) {
            if (current_select_type_block_type_asr) {
                type = current_select_type_block_type_asr;
            } else if (!current_select_type_block_der_type.empty()) {
                type = ASRUtils::make_StructType_t_util(
                                    al, type->base.loc,
                                    current_scope->resolve_symbol(current_select_type_block_der_type));
            }
        }
        if (ASR::is_a<ASR::Integer_t>(*type)) {
            res += "I";
            res += std::to_string(ASRUtils::extract_kind_from_ttype_t(type));
        } else if (ASR::is_a<ASR::UnsignedInteger_t>(*type)) {
            res += "U";
            res += std::to_string(ASRUtils::extract_kind_from_ttype_t(type));
        } else if (ASR::is_a<ASR::Real_t>(*type)) {
            res += "R";
            res += std::to_string(ASRUtils::extract_kind_from_ttype_t(type));
        } else if (ASR::is_a<ASR::String_t>(*type)) {
            res += "S-";
            ASR::String_t* str_type = ASR::down_cast<ASR::String_t>(type);
            if(str_type->m_physical_type == ASR::DescriptorString) res += "DESC";
            else if(str_type->m_physical_type == ASR::CChar) res +="CCHAR";
            else throw LCompilersException("Unhandled string physical type");
            int len;
            if(ASRUtils::extract_value(str_type->m_len, len)){res += "-" + std::to_string(len);}
        } else if (ASR::is_a<ASR::Complex_t>(*type)){
            res += "{R" + std::to_string(ASRUtils::extract_kind_from_ttype_t(type))+
                    ",R" + std::to_string(ASRUtils::extract_kind_from_ttype_t(type))+
                    "}";
        } else if (ASR::is_a<ASR::Array_t>(*type)) {
            // push array size only if it's not a struct member.
            if(in_struct && ASRUtils::is_fixed_size_array(type)){
                res += std::to_string(ASRUtils::get_fixed_size_of_array(type));
            } else if (in_struct && !ASRUtils::is_fixed_size_array(type)){
                // TODO : Throw a semantic error instead.
                throw CodeGenError("Can't print type variable with dynamic array member");
            }
            res += "[";
            res += SerializeType(expr, ASR::down_cast<ASR::Array_t>(type)->m_type, in_struct);
            res += "]";
        } else if (ASR::is_a<ASR::StructType_t>(*type) && !ASRUtils::is_class_type(type)) {
            res += "(";
            if (ASRUtils::is_unlimited_polymorphic_type(expr)) {
                res += serialize_structType_symbols(current_scope->resolve_symbol(current_select_type_block_der_type));
            } else {
                res += serialize_structType_symbols(ASRUtils::symbol_get_past_external(ASRUtils::get_struct_sym_from_struct_expr(expr)));
            }
            res += ")";
        } else if (ASR::is_a<ASR::Logical_t>(*type)) {
            res += "L";
        } else if(ASR::is_a<ASR::CPtr_t>(*type)){
            res += "CPtr";
        } else {
            throw CodeGenError("Printing support is not available for `" +
                ASRUtils::type_to_str_fortran(type) + "` type.");
        }
        return res;
    }
    // Serializes the types of a list of expressions.
    llvm::Value* SerializeExprTypes(ASR::expr_t** args, size_t n_args){
        std::string serialization_res = "";
        for (size_t i=0; i<n_args; i++) {
            serialization_res += SerializeType(args[i], ASRUtils::expr_type(args[i]), false);
            if(i != n_args-1){
                serialization_res += ",";
            }
        }
        return builder->CreateGlobalStringPtr(serialization_res, "serialization_info");
    }

    void compute_fmt_specifier_and_arg(std::vector<std::string> &fmt,
        std::vector<llvm::Value *> &args, ASR::expr_t *v, const Location &loc) {
        int64_t ptr_loads_copy = ptr_loads;
        int reduce_loads = 0;
        ptr_loads = 2;
        if( ASR::is_a<ASR::Var_t>(*v) ) {
            ASR::Variable_t* var = ASRUtils::EXPR2VAR(v);
            reduce_loads = var->m_intent == ASRUtils::intent_in;
            if( LLVM::is_llvm_pointer(*var->m_type) ) {
                ptr_loads = 1;
            }
        }

        ptr_loads = ptr_loads - reduce_loads;
        lookup_enum_value_for_nonints = true;
        ASR::ttype_t *t = ASRUtils::expr_type(v);
        if(ASRUtils::is_character(*t)){
            tmp = get_string_data(v);
        } else {
            this->visit_expr_wrapper(v, true);
        }
        lookup_enum_value_for_nonints = false;
        ptr_loads = ptr_loads_copy;

        if ((t->type == ASR::ttypeType::CPtr && !ASRUtils::is_array(t)) ||
            (t->type == ASR::ttypeType::Pointer &&
                (ASR::is_a<ASR::Var_t>(*v) || ASR::is_a<ASR::GetPointer_t>(*v))
                && !ASRUtils::is_array(t) && !ASRUtils::is_character(*t))
            ) {
            fmt.push_back("%lld");
            llvm::Value* d = builder->CreatePtrToInt(tmp, llvm_utils->getIntType(8, false));
            args.push_back(d);
            return ;
        }

        load_non_array_non_character_pointers(v, ASRUtils::expr_type(v), tmp);
        if (ASRUtils::is_array(t) && ASRUtils::is_allocatable(t)) {
            llvm::Type *llvm_type = llvm_utils->get_type_from_ttype_t_util(v,
                ASRUtils::extract_type(t), module.get());
            tmp = llvm_utils->CreateLoad2(llvm_type->getPointerTo(), tmp);
        }
        t = ASRUtils::extract_type(t);
        int a_kind = ASRUtils::extract_kind_from_ttype_t(t);

        if (ASRUtils::is_integer(*t)) {
            switch( a_kind ) {
                case 1 : {
                    fmt.push_back("%hhi");
                    break;
                }
                case 2 : {
                    fmt.push_back("%hi");
                    break;
                }
                case 4 : {
                    fmt.push_back("%d");
                    break;
                }
                case 8 : {
                    fmt.push_back("%lld");
                    break;
                }
                default: {
                    throw CodeGenError(R"""(Printing support is available only
                                        for 8, 16, 32, and 64 bit integer kinds.)""",
                                        loc);
                }
            }
            args.push_back(tmp);
        } else if (ASRUtils::is_unsigned_integer(*t)) {
            switch( a_kind ) {
                case 1 : {
                    fmt.push_back("%hhu");
                    break;
                }
                case 2 : {
                    fmt.push_back("%hu");
                    break;
                }
                case 4 : {
                    fmt.push_back("%u");
                    break;
                }
                case 8 : {
                    fmt.push_back("%llu");
                    break;
                }
                default: {
                    throw CodeGenError(R"""(Printing support is available only
                                        for 8, 16, 32, and 64 bit unsigned integer kinds.)""",
                                        loc);
                }
            }
            args.push_back(tmp);
        } else if (ASRUtils::is_real(*t)) {
            switch( a_kind ) {
                case 4 : {
                        fmt.push_back("%13.8e");
                    break;
                }
                case 8 : {
                        fmt.push_back("%23.17e");
                    break;
                }
                default: {
                    throw CodeGenError(R"""(Printing support is available only
                                        for 32, and 64 bit real kinds.)""",
                                        loc);
                }
            }
            args.push_back(tmp);
        } else if (t->type == ASR::ttypeType::String) {
            fmt.push_back("%s");
            args.push_back(tmp);
        } else if (ASRUtils::is_logical(*t)) {
            fmt.push_back("%s");
            args.push_back(tmp);
        } else if (ASRUtils::is_complex(*t)) {
            switch( a_kind ) {
                case 4 : {
                    fmt.push_back("(%f,%f)");
                    break;
                }
                case 8 : {
                    fmt.push_back("(%lf,%lf)");
                    break;
                }
                default: {
                    throw CodeGenError(R"""(Printing support is available only
                                        for 32, and 64 bit complex kinds.)""",
                                        loc);
                }
            }
            args.push_back(tmp);
        } else if (t->type == ASR::ttypeType::CPtr) {
            fmt.push_back("%lld");
            args.push_back(tmp);
        } else if (t->type == ASR::ttypeType::EnumType) {
            // TODO: Use recursion to generalise for any underlying type in enum
            fmt.push_back("%d");
            args.push_back(tmp);
        } else {
            throw CodeGenError("Printing support is not available for `" +
                ASRUtils::type_to_str_fortran(t) + "` type.", loc);
        }
    }

    void handle_print(ASR::expr_t* arg, ASR::expr_t* end_expr) {
        std::vector<llvm::Value *> args;
        args.push_back(nullptr); // reserve space for fmt_str
        std::vector<std::string> fmt;
        llvm::Value* end{};
        if(end_expr == nullptr){
            end = builder->CreateGlobalStringPtr("\n");
        } else {
            end = get_string_data(end_expr);
        }
        compute_fmt_specifier_and_arg(fmt, args, arg, arg->base.loc);
        fmt.push_back("%s");
        args.push_back(end);
        std::string fmt_str;
        for (size_t i=0; i<fmt.size(); i++) {
            fmt_str += fmt[i];
        }
        llvm::Value *fmt_ptr = builder->CreateGlobalStringPtr(fmt_str);
        args[0] = fmt_ptr;
        printf(context, *module, *builder, args);
    }

    void construct_stop(llvm::Value* exit_code, std::string stop_msg, ASR::expr_t* stop_code, Location loc) {
        std::string fmt_str;
        std::vector<std::string> fmt;
        std::vector<llvm::Value*> args;
        args.push_back(nullptr); // reserve space for fmt_str
        ASR::ttype_t *str_type_len_msg = ASRUtils::TYPE(ASR::make_String_t(
                    al, loc, 1,
                    ASRUtils::EXPR(ASR::make_IntegerConstant_t(al, loc, stop_msg.size(),
                        ASRUtils::TYPE(ASR::make_Integer_t(al, loc, 4)))),
                    ASR::string_length_kindType::ExpressionLength,
                    ASR::string_physical_typeType::DescriptorString));
        ASR::expr_t* STOP_MSG = ASRUtils::EXPR(ASR::make_StringConstant_t(al, loc,
            s2c(al, stop_msg), str_type_len_msg));
        ASR::ttype_t *str_type_len_1 = ASRUtils::TYPE(ASR::make_String_t(
                al, loc, 1,
                ASRUtils::EXPR(ASR::make_IntegerConstant_t(al, loc, 1,
                    ASRUtils::TYPE(ASR::make_Integer_t(al, loc, 4)))),
                ASR::string_length_kindType::ExpressionLength,
                ASR::string_physical_typeType::DescriptorString));
        ASR::expr_t* NEWLINE = ASRUtils::EXPR(ASR::make_StringConstant_t(al, loc,
            s2c(al, "\n"), str_type_len_1));
        compute_fmt_specifier_and_arg(fmt, args, STOP_MSG, loc);
        if (stop_code) {
            ASR::expr_t* SPACE = ASRUtils::EXPR(ASR::make_StringConstant_t(al, loc,
                s2c(al, " "), str_type_len_1));
            compute_fmt_specifier_and_arg(fmt, args, SPACE, loc);
            compute_fmt_specifier_and_arg(fmt, args, stop_code, loc);
        }
        compute_fmt_specifier_and_arg(fmt, args, NEWLINE, loc);

        for (auto ch:fmt) {
            fmt_str += ch;
        }

        llvm::Value *fmt_ptr = builder->CreateGlobalStringPtr(fmt_str);
        args[0] = fmt_ptr;
        print_error(context, *module, *builder, args);

        if (stop_code && is_a<ASR::Integer_t>(*ASRUtils::expr_type(stop_code))) {
            this->visit_expr(*stop_code);
            exit_code = tmp;
        }
        exit(context, *module, *builder, exit_code);
    }

    void visit_Stop(const ASR::Stop_t &x) {
        if (compiler_options.emit_debug_info) {
            debug_emit_loc(x);
            if (x.m_code && is_a<ASR::Integer_t>(*ASRUtils::expr_type(x.m_code))) {
                llvm::Value *fmt_ptr = builder->CreateGlobalStringPtr(infile);
                llvm::Value *fmt_ptr1 = llvm::ConstantInt::get(context, llvm::APInt(
                    1, compiler_options.use_colors));
                this->visit_expr(*x.m_code);
                llvm::Value *test = builder->CreateICmpNE(tmp, builder->getInt32(0));
                llvm_utils->create_if_else(test, [=]() {
                    call_print_stacktrace_addresses(context, *module, *builder,
                        {fmt_ptr, fmt_ptr1});
                }, [](){});
            }
        }

        int exit_code_int = 0;
        llvm::Value *exit_code = llvm::ConstantInt::get(context,
                llvm::APInt(32, exit_code_int));
        construct_stop(exit_code, "STOP", x.m_code, x.base.base.loc);
    }

    void visit_ErrorStop(const ASR::ErrorStop_t &x) {
        if (compiler_options.emit_debug_info) {
            debug_emit_loc(x);
            llvm::Value *fmt_ptr = builder->CreateGlobalStringPtr(infile);
            llvm::Value *fmt_ptr1 = llvm::ConstantInt::get(context, llvm::APInt(
                1, compiler_options.use_colors));
            call_print_stacktrace_addresses(context, *module, *builder,
                {fmt_ptr, fmt_ptr1});
        }

        int exit_code_int = 1;
        llvm::Value *exit_code = llvm::ConstantInt::get(context,
                llvm::APInt(32, exit_code_int));
        construct_stop(exit_code, "ERROR STOP", x.m_code, x.base.base.loc);
    }

    template <typename T>
    inline void set_func_subrout_params(T* func_subrout, ASR::abiType& x_abi,
                                        std::uint32_t& m_h, ASR::Variable_t*& orig_arg,
                                        std::string& orig_arg_name, ASR::intentType& arg_intent,
                                        size_t arg_idx) {
        m_h = get_hash((ASR::asr_t*)func_subrout);
        if( ASR::is_a<ASR::Var_t>(*func_subrout->m_args[arg_idx]) ) {
            ASR::Var_t* arg_var = ASR::down_cast<ASR::Var_t>(func_subrout->m_args[arg_idx]);
            ASR::symbol_t* arg_sym = symbol_get_past_external(arg_var->m_v);
            if( ASR::is_a<ASR::Variable_t>(*arg_sym) ) {
                orig_arg = ASR::down_cast<ASR::Variable_t>(arg_sym);
                orig_arg_name = orig_arg->m_name;
                arg_intent = orig_arg->m_intent;
            }
        }
        x_abi = ASRUtils::get_FunctionType(func_subrout)->m_abi;
    }


    template <typename T>
    std::vector<llvm::Value*> convert_call_args(const T &x, bool is_method) {
        std::vector<llvm::Value *> args;
        for (size_t i=0; i<x.n_args; i++) {
            ASR::symbol_t* func_subrout = symbol_get_past_external(x.m_name);
            ASR::abiType x_abi = (ASR::abiType) 0;
            ASR::intentType orig_arg_intent = ASR::intentType::Unspecified;
            std::uint32_t m_h;
            ASR::Variable_t *orig_arg = nullptr;
            std::string orig_arg_name = "";
            if( func_subrout->type == ASR::symbolType::Function ) {
                ASR::Function_t* func = down_cast<ASR::Function_t>(func_subrout);
                set_func_subrout_params(func, x_abi, m_h, orig_arg, orig_arg_name, orig_arg_intent, i + is_method);
            } else if( func_subrout->type == ASR::symbolType::StructMethodDeclaration ) {
                ASR::StructMethodDeclaration_t* clss_proc = ASR::down_cast<ASR::StructMethodDeclaration_t>(func_subrout);
                if( clss_proc->m_proc->type == ASR::symbolType::Function ) {
                    ASR::Function_t* func = down_cast<ASR::Function_t>(clss_proc->m_proc);
                    set_func_subrout_params(func, x_abi, m_h, orig_arg, orig_arg_name, orig_arg_intent, i + is_method);
                    func_subrout = clss_proc->m_proc;
                }
            } else if( func_subrout->type == ASR::symbolType::Variable ) {
                ASR::Variable_t* v = down_cast<ASR::Variable_t>(func_subrout);
                ASR::Function_t* func = down_cast<ASR::Function_t>(ASRUtils::symbol_get_past_external(v->m_type_declaration));
                func_subrout = ASRUtils::symbol_get_past_external(v->m_type_declaration);
                set_func_subrout_params(func, x_abi, m_h, orig_arg, orig_arg_name, orig_arg_intent, i + is_method);
            } else {
                LCOMPILERS_ASSERT(false)
            }

            if( x.m_args[i].m_value == nullptr ) {
                LCOMPILERS_ASSERT(orig_arg != nullptr);
                llvm::Type* llvm_orig_arg_type = llvm_utils->get_type_from_ttype_t_util(ASRUtils::EXPR(ASR::make_Var_t(
                    al, orig_arg->base.base.loc, &orig_arg->base)),
                    orig_arg->m_type, module.get());
                llvm::Value* llvm_arg = llvm_utils->CreateAlloca(*builder, llvm_orig_arg_type);
                args.push_back(llvm_arg);
                continue ;
            }
            if(ASRUtils::is_character(*ASRUtils::expr_type(x.m_args[i].m_value)) &&
                ASRUtils::is_character(*ASRUtils::extract_type(orig_arg->m_type))){
                LCOMPILERS_ASSERT_MSG(
                    ASRUtils::is_character_phsyical_types_matched(
                        ASRUtils::extract_type(ASRUtils::expr_type(x.m_args[i].m_value)),
                        ASRUtils::extract_type(orig_arg->m_type)),
                    "Unmatching String Physical Types");
            }
            if (ASR::is_a<ASR::Var_t>(*x.m_args[i].m_value)) {
                ASR::symbol_t* var_sym = ASRUtils::symbol_get_past_external(
                    ASR::down_cast<ASR::Var_t>(x.m_args[i].m_value)->m_v);
                if (ASR::is_a<ASR::Variable_t>(*var_sym)) {
                    ASR::Variable_t *arg = EXPR2VAR(x.m_args[i].m_value);
                    uint32_t h = get_hash((ASR::asr_t*)arg);
                    if (llvm_symtab.find(h) != llvm_symtab.end()) {
                        if (llvm_symtab_deep_copy.find(h) != llvm_symtab_deep_copy.end()) {
                            tmp = llvm_symtab_deep_copy[h];
                        } else {
                            tmp = llvm_symtab[h];
                        }
                        if( !ASRUtils::is_array(arg->m_type) ) {

                            if (x_abi == ASR::abiType::Source && ASR::is_a<ASR::CPtr_t>(*arg->m_type)) {
                                if ( orig_arg_intent != ASRUtils::intent_out &&
                                        arg->m_intent == intent_local ) {
                                    // Local variable of type
                                    // CPtr is a void**, so we
                                    // have to load it
                                    tmp = llvm_utils->CreateLoad(tmp);
                                }
                            } else if ( x_abi == ASR::abiType::BindC ) {
                                if (orig_arg->m_abi == ASR::abiType::BindC && orig_arg->m_value_attr) {
                                    ASR::ttype_t* arg_type = arg->m_type;
                                    llvm::Type* arg_llvm_type = llvm_utils->get_type_from_ttype_t_util(ASRUtils::EXPR(ASR::make_Var_t(
                                        al, arg->base.base.loc, &arg->base)), arg_type, module.get());
                                    if (ASR::is_a<ASR::Complex_t>(*arg_type)) {
                                        if (!startswith(compiler_options.target, "wasm")) {
                                            int c_kind = ASRUtils::extract_kind_from_ttype_t(arg_type);
                                            if (c_kind == 4) {
                                                if (compiler_options.platform == Platform::Windows) {
                                                    // tmp is {float, float}*
                                                    // type_fx2 is i64
                                                    llvm::Type* type_fx2 = llvm::Type::getInt64Ty(context);
                                                    tmp = llvm_utils->CreateLoad2(type_fx2, tmp);
                                                } else if (compiler_options.platform == Platform::macOS_ARM) {
                                                    // tmp is {float, float}*
                                                    // type_fx2 is [2 x float]
                                                    llvm::Type* type_fx2 = llvm::ArrayType::get(llvm::Type::getFloatTy(context), 2);
                                                    // we bitcast complex_type_4 to [2 x float]*
                                                    llvm::Value *complex_as_array_ptr = builder->CreateBitCast(tmp, type_fx2->getPointerTo());
                                                    tmp = llvm_utils->CreateLoad2(type_fx2, complex_as_array_ptr);
                                                } else {
                                                    // tmp is {float, float}*
                                                    // type_fx2 is <2 x float>
                                                    llvm::Type* type_fx2 = FIXED_VECTOR_TYPE::get(llvm::Type::getFloatTy(context), 2);
                                                    tmp = llvm_utils->CreateLoad2(type_fx2, tmp);
                                                }
                                            } else {
                                                LCOMPILERS_ASSERT(c_kind == 8)
                                                if (compiler_options.platform == Platform::Windows) {
                                                    // 128 bit aggregate type is passed by reference
                                                } else {
                                                    // Pass by value
                                                    tmp = llvm_utils->CreateLoad2(arg_llvm_type, tmp);
                                                }
                                            }
                                        }
                                    } else if (is_a<ASR::CPtr_t>(*arg_type)) {
                                        if ( arg->m_intent == intent_local ||
                                                arg->m_intent == ASRUtils::intent_out) {
                                            // Local variable or Dummy out argument
                                            // of type CPtr is a void**, so we
                                            // have to load it
                                            tmp = llvm_utils->CreateLoad(tmp);
                                        }
                                    } else {
                                        if (!arg->m_value_attr && !ASR::is_a<ASR::String_t>(*arg_type)) {
                                            // Dereference the pointer argument (unless it is a CPtr)
                                            // to pass by value
                                            // E.g.:
                                            // i32* -> i32
                                            // {double,double}* -> {double,double}
                                            tmp = llvm_utils->CreateLoad2(arg_llvm_type, tmp);
                                        }
                                    }
                                }
                                if (!orig_arg->m_value_attr && arg->m_value_attr) {
                                    llvm::Type *target_type = tmp->getType();
                                    // Create alloca to get a pointer, but do it
                                    // at the beginning of the function to avoid
                                    // using alloca inside a loop, which would
                                    // run out of stack
                                    llvm::AllocaInst *target = llvm_utils->CreateAlloca(
                                        target_type, nullptr, "call_arg_value_ptr");
                                    builder->CreateStore(tmp, target);
                                    tmp = target;
                                }
                            } else {
                                if( arg->m_intent == intent_local && ASR::is_a<ASR::FunctionType_t>(*arg->m_type) ) {
                                    // (FunctionType**) --> (FunctionType*)
                                    bool pass_by_value = true;
                                    if ( ASR::is_a<ASR::Function_t>(*func_subrout) ) {
                                        ASR::Function_t* func = ASR::down_cast<ASR::Function_t>(func_subrout);
                                        if ( ASR::is_a<ASR::Var_t>(*func->m_args[i]) ){
                                            ASR::symbol_t* func_var_sym = ASRUtils::symbol_get_past_external(
                                                ASR::down_cast<ASR::Var_t>(func->m_args[i])->m_v);
                                            if ( ASR::is_a<ASR::Variable_t>(*func_var_sym) ) {
                                                ASR::Variable_t* func_variable = ASR::down_cast<ASR::Variable_t>(func_var_sym);
                                                if ( func_variable->m_intent == ASRUtils::intent_inout || func_variable->m_intent == ASRUtils::intent_out ) {
                                                    pass_by_value = false;
                                                }
                                            }
                                        }
                                    }

                                    if ( pass_by_value ) {
                                        // Pass-by-Value
                                        tmp = llvm_utils->CreateLoad2(llvm_utils->get_type_from_ttype_t_util(ASRUtils::EXPR(ASR::make_Var_t(
                                            al, arg->base.base.loc, &arg->base)), arg->m_type, module.get()), tmp);
                                    }
                                }
                                if( orig_arg &&
                                    !LLVM::is_llvm_pointer(*orig_arg->m_type) &&
                                    LLVM::is_llvm_pointer(*arg->m_type) &&
                                    !ASRUtils::is_character(*arg->m_type) &&
                                    !ASRUtils::is_class_type(ASRUtils::type_get_past_allocatable_pointer(arg->m_type))) {
                                    // TODO: Remove call to ASRUtils::check_equal_type
                                    // pass(rhs) is not respected in integration_tests/class_08.f90
                                    tmp = llvm_utils->CreateLoad(tmp);
                                }

                                if (ASRUtils::is_class_type(
                                        ASRUtils::type_get_past_allocatable(arg->m_type))
                                    && !ASRUtils::is_unlimited_polymorphic_type(x.m_args[i].m_value)
                                    && !ASRUtils::is_class_type(
                                        ASRUtils::type_get_past_allocatable(orig_arg->m_type))) {
                                    tmp = convert_class_to_type(x.m_args[i].m_value, ASRUtils::EXPR(ASR::make_Var_t(
                                        al, orig_arg->base.base.loc, &orig_arg->base)), orig_arg->m_type, tmp);
                                }
                            }
                        } else {
                            if( orig_arg &&
                                !LLVM::is_llvm_pointer(*orig_arg->m_type) &&
                                LLVM::is_llvm_pointer(*arg->m_type) ) {
                                tmp = llvm_utils->CreateLoad(tmp);
                            }
                        }
                    } else {
                        if ( arg->m_type_declaration && ASR::is_a<ASR::Function_t>(
                                *ASRUtils::symbol_get_past_external(arg->m_type_declaration)) ) {
                            ASR::Function_t* fn = ASR::down_cast<ASR::Function_t>(
                                symbol_get_past_external(arg->m_type_declaration));
                            uint32_t h = get_hash((ASR::asr_t*)fn);
                            if (ASRUtils::get_FunctionType(fn)->m_deftype == ASR::deftypeType::Implementation) {
                                LCOMPILERS_ASSERT(llvm_symtab_fn.find(h) != llvm_symtab_fn.end());
                                tmp = llvm_symtab_fn[h];
                            } else {
                                // Must be an argument/chained procedure pass
                                tmp = llvm_symtab_fn_arg[h];
                            }
                        } else {
                            if (arg->m_value == nullptr) {
                                throw CodeGenError(std::string(arg->m_name) + " isn't defined in any scope.");
                            }
                            this->visit_expr_wrapper(arg->m_value, true);
                            if( x_abi != ASR::abiType::BindC &&
                                !ASR::is_a<ASR::ArrayConstant_t>(*arg->m_value) ) {
                                llvm::AllocaInst *target = llvm_utils->CreateAlloca(
                                    llvm_utils->get_type_from_ttype_t_util(ASRUtils::EXPR(ASR::make_Var_t(
                                    al, arg->base.base.loc, &arg->base)), arg->m_type, module.get()),
                                    nullptr, "call_arg_value");
                                builder->CreateStore(tmp, target);
                                tmp = target;
                            }
                        }
                    }
                    if (orig_arg &&
                        LLVM::is_llvm_pointer(*orig_arg->m_type) &&
                        !LLVM::is_llvm_pointer(*arg->m_type)) {
                        llvm::Value* ptr_to_tmp = llvm_utils->CreateAlloca(*builder, tmp->getType());
                        builder->CreateStore(tmp, ptr_to_tmp);
                        tmp = ptr_to_tmp;
                    }
                } else if (ASR::is_a<ASR::Function_t>(*var_sym)) {
                    ASR::Function_t* fn = ASR::down_cast<ASR::Function_t>(var_sym);
                    uint32_t h = get_hash((ASR::asr_t*)fn);
                    if (ASRUtils::get_FunctionType(fn)->m_deftype == ASR::deftypeType::Implementation) {
                        LCOMPILERS_ASSERT(llvm_symtab_fn.find(h) != llvm_symtab_fn.end());
                        tmp = llvm_symtab_fn[h];
                    } else if (llvm_symtab_fn_arg.find(h) == llvm_symtab_fn_arg.end() &&
                                ASR::is_a<ASR::Function_t>(*var_sym) &&
                                ASRUtils::get_FunctionType(fn)->m_deftype == ASR::deftypeType::Interface ) {
                        LCOMPILERS_ASSERT(llvm_symtab_fn.find(h) != llvm_symtab_fn.end());
                        tmp = llvm_symtab_fn[h];
                        LCOMPILERS_ASSERT(tmp != nullptr)
                    } else {
                        // Must be an argument/chained procedure pass
                        LCOMPILERS_ASSERT(llvm_symtab_fn_arg.find(h) != llvm_symtab_fn_arg.end());
                        tmp = llvm_symtab_fn_arg[h];
                        LCOMPILERS_ASSERT(tmp != nullptr)
                    }
                }
            } else if (ASR::is_a<ASR::ArrayPhysicalCast_t>(*x.m_args[i].m_value)) {
                this->visit_expr_wrapper(x.m_args[i].m_value);
            } else if( ASR::is_a<ASR::FunctionType_t>(
                *ASRUtils::type_get_past_pointer(
                    ASRUtils::type_get_past_allocatable(
                        ASRUtils::expr_type(x.m_args[i].m_value)))) ) {
                this->visit_expr_wrapper(x.m_args[i].m_value, true);
            } else {
                ASR::ttype_t* arg_type = expr_type(x.m_args[i].m_value);
                this->visit_expr_wrapper(x.m_args[i].m_value);

                if( x_abi == ASR::abiType::BindC ) {
                    if( (ASR::is_a<ASR::ArrayItem_t>(*x.m_args[i].m_value) &&
                         orig_arg_intent ==  ASR::intentType::In) ||
                        ASR::is_a<ASR::StructInstanceMember_t>(*x.m_args[i].m_value) ||
                        (ASR::is_a<ASR::CPtr_t>(*arg_type) &&
                         ASR::is_a<ASR::StructInstanceMember_t>(*x.m_args[i].m_value)) ) {
                        if( ASR::is_a<ASR::StructInstanceMember_t>(*x.m_args[i].m_value) &&
                            ASRUtils::is_array(arg_type) ) {
                            ASR::dimension_t* arg_m_dims = nullptr;
                            size_t n_dims = ASRUtils::extract_dimensions_from_ttype(arg_type, arg_m_dims);
                            if( !(ASRUtils::is_fixed_size_array(arg_m_dims, n_dims) &&
                                  ASRUtils::expr_abi(x.m_args[i].m_value) == ASR::abiType::BindC) ) {
                                tmp = llvm_utils->CreateLoad(arr_descr->get_pointer_to_data(tmp));
                            } else {
                                tmp = llvm_utils->create_gep(tmp, llvm::ConstantInt::get(
                                        llvm::Type::getInt32Ty(context), llvm::APInt(32, 0)));
                            }
                        } else {
                            tmp = llvm_utils->CreateLoad(tmp);
                        }
                    }
                }
                if (ASRUtils::is_pointer(arg_type) && !ASRUtils::is_array(arg_type) &&
                        ASR::is_a<ASR::StructType_t>(*ASRUtils::extract_type(arg_type)) &&
                        !LLVM::is_llvm_pointer(*orig_arg->m_type)) {
                    llvm::Type *el_type = llvm_utils->get_type_from_ttype_t_util(ASRUtils::EXPR(ASR::make_Var_t(
                    al, orig_arg->base.base.loc, &orig_arg->base)), orig_arg->m_type, module.get());
                    tmp = llvm_utils->CreateLoad2(el_type->getPointerTo(), tmp);
                }
                llvm::Value *value = tmp;
                if (orig_arg_intent == ASR::intentType::In ||
                    orig_arg_intent == ASR::intentType::InOut ||
                    orig_arg_intent == ASR::intentType::Out) {
                    /*
                        For the cases where argument intent is In or InOut or Out, we
                        cannot pass the evaluated value of expression directly to it. Currently
                        the value of `value` is the evaluated value of expression and this is because
                        for every expression we visit, we have a check to prefer the evaluated value.

                        To avoid this problem, we manually set the expr value to nullptr.
                        For example, refer integration_tests/intent_03.f90
                    */
                    // TODO: this is possible in other cases as well, support those.
                    if ( ASR::is_a<ASR::ArrayItem_t>(*x.m_args[i].m_value) ) {
                        ASR::ArrayItem_t* arr_item = ASR::down_cast<ASR::ArrayItem_t>(x.m_args[i].m_value);
                        arr_item->m_value = nullptr;
                        this->visit_expr_wrapper((ASR::expr_t*)arr_item);
                        value = tmp;
                    }
                }
                // TODO: we are getting a warning of uninitialized variable,
                // there might be a bug below.
                llvm::Type *target_type = nullptr;
                ASR::ttype_t* arg_type_ = ASRUtils::type_get_past_array(arg_type);
                switch (arg_type_->type) {
                    case (ASR::ttypeType::Integer) : {
                        int a_kind = down_cast<ASR::Integer_t>(arg_type_)->m_kind;
                        target_type = llvm_utils->getIntType(a_kind);
                        break;
                    }
                    case (ASR::ttypeType::UnsignedInteger) : {
                        int a_kind = down_cast<ASR::UnsignedInteger_t>(arg_type_)->m_kind;
                        target_type = llvm_utils->getIntType(a_kind);
                        break;
                    }
                    case (ASR::ttypeType::Real) : {
                        int a_kind = down_cast<ASR::Real_t>(arg_type_)->m_kind;
                        target_type = llvm_utils->getFPType(a_kind);
                        break;
                    }
                    case (ASR::ttypeType::Complex) : {
                        int a_kind = down_cast<ASR::Complex_t>(arg_type_)->m_kind;
                        target_type = llvm_utils->getComplexType(a_kind);
                        break;
                    }
                    case (ASR::ttypeType::String) : {
                        target_type = llvm_utils->get_StringType(arg_type_);
                        break;
                    }
                    case (ASR::ttypeType::Logical) :
                        target_type = llvm::Type::getInt1Ty(context);
                        break;
                    case (ASR::ttypeType::EnumType) :
                        target_type = llvm::Type::getInt32Ty(context);
                        break;
                    case (ASR::ttypeType::StructType) :
                        break;
                    case (ASR::ttypeType::CPtr) :
                        target_type = llvm::Type::getVoidTy(context)->getPointerTo();
                        break;
                    case ASR::ttypeType::Allocatable:
                    case (ASR::ttypeType::Pointer) : {
                        ASR::ttype_t* type_ = ASRUtils::get_contained_type(arg_type);
                        target_type = llvm_utils->get_type_from_ttype_t_util(x.m_args[i].m_value, type_, module.get());
                        target_type = target_type->getPointerTo();
                        break;
                    }
                    case (ASR::ttypeType::List) : {
                        target_type = llvm_utils->get_type_from_ttype_t_util(x.m_args[i].m_value, arg_type_, module.get());
                        break ;
                    }
                    case (ASR::ttypeType::Tuple) : {
                        target_type = llvm_utils->get_type_from_ttype_t_util(x.m_args[i].m_value, arg_type_, module.get());
                        break ;
                    }
                    case (ASR::ttypeType::FunctionType): {
                        target_type = llvm_utils->get_type_from_ttype_t_util(x.m_args[i].m_value, arg_type_, module.get());
                        break;
                    }
                    default :
                        throw CodeGenError("Type " + ASRUtils::type_to_str_fortran(arg_type) + " not implemented yet.");
                }
                if( ASR::is_a<ASR::EnumValue_t>(*x.m_args[i].m_value) ) {
                    target_type = llvm::Type::getInt32Ty(context);
                }
                if (ASR::is_a<ASR::StructType_t>(*arg_type) && !ASRUtils::is_class_type(arg_type)) {
                    tmp = value;
                } else if(ASR::is_a<ASR::String_t>(*arg_type)){
                    tmp = value;
                }else {
                    bool use_value = false;
                    ASR::Variable_t *orig_arg = nullptr;
                    if( func_subrout->type == ASR::symbolType::Function ) {
                        ASR::Function_t* func = down_cast<ASR::Function_t>(func_subrout);
                        orig_arg = EXPR2VAR(func->m_args[i]);
                    } else {
                        LCOMPILERS_ASSERT(false)
                    }
                    if (orig_arg->m_abi == ASR::abiType::BindC
                        && orig_arg->m_value_attr) {
                        use_value = true;
                    }
                    if (ASR::is_a<ASR::ArrayItem_t>(*x.m_args[i].m_value)) {
                        use_value = true;
                    }
                    if (!use_value) {
                        // Create alloca to get a pointer, but do it
                        // at the beginning of the function to avoid
                        // using alloca inside a loop, which would
                        // run out of stack
                        if ((ASR::is_a<ASR::ArrayItem_t>(*x.m_args[i].m_value)
                                || (ASR::is_a<ASR::StructInstanceMember_t>(*x.m_args[i].m_value)
                                    && ((!ASRUtils::is_allocatable(orig_arg->m_type)
                                        && ASRUtils::is_allocatable(arg_type))
                                        || !ASRUtils::is_allocatable(arg_type))
                                    && (ASRUtils::is_array(arg_type)
                                        || ASR::is_a<ASR::CPtr_t>(
                                            *ASRUtils::expr_type(x.m_args[i].m_value))))
                                || (ASR::is_a<ASR::StructInstanceMember_t>(*x.m_args[i].m_value)
                                    && ASRUtils::is_allocatable(arg_type)
                                    && !ASRUtils::is_allocatable(orig_arg->m_type)
                                    && (orig_arg->m_intent == ASR::intentType::Out
                                        || orig_arg->m_intent == ASR::intentType::InOut)))
                            && value->getType()->isPointerTy()
                            && !ASRUtils::is_character(*arg_type)) {
                                if (ASRUtils::is_class_type(
                                        ASRUtils::type_get_past_allocatable(arg_type))
                                    && !ASRUtils::is_class_type(
                                        ASRUtils::type_get_past_allocatable(orig_arg->m_type))) {
                                    value = convert_class_to_type(x.m_args[i].m_value, ASRUtils::EXPR(ASR::make_Var_t(
                                        al, orig_arg->base.base.loc, &orig_arg->base)),
                                        orig_arg->m_type, value);
                                } else {
                                    if (compiler_options.po.realloc_lhs) {
                                        check_and_allocate(x.m_args[i].m_value);
                                    }
                                    value = llvm_utils->CreateLoad(value);
                                }
                        }
                        if( !ASR::is_a<ASR::CPtr_t>(*arg_type) &&
                            !(orig_arg && !LLVM::is_llvm_pointer(*orig_arg->m_type) &&
                            LLVM::is_llvm_pointer(*arg_type)) &&
                            !ASRUtils::is_character(*arg_type) &&
                            (!ASR::is_a<ASR::StructInstanceMember_t>(*x.m_args[i].m_value) || ASRUtils::is_value_constant(x.m_args[i].m_value)) ) {
                            llvm::AllocaInst *target = llvm_utils->CreateAlloca(
                                target_type, nullptr, "call_arg_value");
                            if( ASR::is_a<ASR::Tuple_t>(*arg_type) ||
                                ASR::is_a<ASR::List_t>(*arg_type) ) {
                                    llvm_utils->deepcopy(x.m_args[i].m_value, value, target, arg_type, arg_type, module.get(), name2memidx);
                            } else {
                                builder->CreateStore(value, target);
                            }
                            tmp = target;
                        } else {
                            tmp = value;
                        }
                    }
                }
            }

            // To avoid segmentation faults when original argument
            // is not a ASR::Variable_t like callbacks.
            if( orig_arg && !(ASR::is_a<ASR::Pointer_t>(*expr_type(x.m_args[i].m_value)) && ASRUtils::is_class_type(
                ASRUtils::type_get_past_array(
                    ASRUtils::type_get_past_allocatable(
                        ASRUtils::type_get_past_pointer(
                            ASRUtils::expr_type(x.m_args[i].m_value)))))) ) {

                tmp = convert_to_polymorphic_arg(x.m_args[i].m_value, tmp,
                    ASRUtils::EXPR(ASR::make_Var_t(al, orig_arg->base.base.loc, &orig_arg->base)),
                    ASRUtils::type_get_past_allocatable(
                        ASRUtils::type_get_past_pointer(orig_arg->m_type)),
                    ASRUtils::type_get_past_allocatable(
                        ASRUtils::type_get_past_pointer(ASRUtils::expr_type(x.m_args[i].m_value))));
            }

            args.push_back(tmp);
        }
        return args;
    }

    llvm::Value *convert_class_to_type(ASR::expr_t *arg, ASR::expr_t* dest_arg, ASR::ttype_t *dest_type, llvm::Value *class_value) {
        // if the required argument is of struct type, we need to pass
        // in the struct pointer to it
        llvm::Value *value = nullptr;
        ASR::ttype_t *arg_type = ASRUtils::expr_type(arg);
        LCOMPILERS_ASSERT(ASRUtils::is_class_type(ASRUtils::type_get_past_allocatable(arg_type)) &&
                          !ASRUtils::is_class_type(ASRUtils::type_get_past_allocatable(dest_type)));
        if (compiler_options.po.realloc_lhs && ASRUtils::is_allocatable(arg_type)) {
            check_and_allocate(arg, dest_arg, dest_type);
        }
        ASR::ttype_t* struct_type = ASRUtils::make_StructType_t_util(
                                        al, arg_type->base.loc,
                                        ASRUtils::symbol_get_past_external(ASRUtils::get_struct_sym_from_struct_expr(arg)));
        llvm::Type* llvm_struct_type = llvm_utils->get_type_from_ttype_t_util(arg, struct_type,
                module.get())->getPointerTo();
        llvm::Value* struct_ptr = llvm_utils->create_gep2(
                                llvm_utils->get_type_from_ttype_t_util(arg,
                                    ASRUtils::type_get_past_allocatable(arg_type),
                                    module.get()),
                                class_value, 1);

        if (ASRUtils::is_class_type(
                ASRUtils::type_get_past_allocatable(arg_type))
            && !ASR::is_a<ASR::Allocatable_t>(*dest_type)) {
            value = llvm_utils->CreateLoad2(llvm_struct_type, struct_ptr);
            value = builder->CreateBitCast(value, llvm_utils->get_type_from_ttype_t_util(dest_arg, dest_type, module.get())->getPointerTo());
        } else {
            value = struct_ptr;
        }

        return value;
    }

    void generate_flip_sign(ASR::call_arg_t* m_args) {
        this->visit_expr_wrapper(m_args[0].m_value, true);
        llvm::Value* signal = tmp;
        LCOMPILERS_ASSERT(m_args[1].m_value->type == ASR::exprType::Var);
        ASR::Var_t* asr_var = ASR::down_cast<ASR::Var_t>(m_args[1].m_value);
        ASR::Variable_t* asr_variable = ASR::down_cast<ASR::Variable_t>(asr_var->m_v);
        uint32_t x_h = get_hash((ASR::asr_t*)asr_variable);
        llvm::Value* variable = llvm_symtab[x_h];
        // variable = xor(shiftl(int(Nd), 63), variable)
        ASR::ttype_t* signal_type = ASRUtils::expr_type(m_args[0].m_value);
        int signal_kind = ASRUtils::extract_kind_from_ttype_t(signal_type);
        llvm::Value* num_shifts = llvm::ConstantInt::get(context, llvm::APInt(32, signal_kind * 8 - 1));
        llvm::Value* shifted_signal = builder->CreateShl(signal, num_shifts);
        llvm::Value* int_var = builder->CreateBitCast(llvm_utils->CreateLoad(variable), shifted_signal->getType());
        tmp = builder->CreateXor(shifted_signal, int_var);
        llvm::Type* variable_type = llvm_utils->get_type_from_ttype_t_util(&asr_var->base, asr_variable->m_type, module.get());
        tmp = builder->CreateBitCast(tmp, variable_type);
    }

    void generate_fma(ASR::call_arg_t* m_args) {
        this->visit_expr_wrapper(m_args[0].m_value, true);
        llvm::Value* a = tmp;
        this->visit_expr_wrapper(m_args[1].m_value, true);
        llvm::Value* b = tmp;
        this->visit_expr_wrapper(m_args[2].m_value, true);
        llvm::Value* c = tmp;
        tmp = builder->CreateIntrinsic(llvm::Intrinsic::fma,
                {a->getType()},
                {b, c, a});
    }

    void generate_sign_from_value(ASR::call_arg_t* m_args) {
        this->visit_expr_wrapper(m_args[0].m_value, true);
        llvm::Value* arg0 = tmp;
        this->visit_expr_wrapper(m_args[1].m_value, true);
        llvm::Value* arg1 = tmp;
        llvm::Type* common_llvm_type = arg0->getType();
        ASR::ttype_t *arg1_type = ASRUtils::expr_type(m_args[1].m_value);
        uint64_t kind = ASRUtils::extract_kind_from_ttype_t(arg1_type);
        llvm::Value* num_shifts = llvm::ConstantInt::get(context, llvm::APInt(kind * 8, kind * 8 - 1));
        llvm::Value* shifted_one = builder->CreateShl(llvm::ConstantInt::get(context, llvm::APInt(kind * 8, 1)), num_shifts);
        arg1 = builder->CreateBitCast(arg1, shifted_one->getType());
        arg0 = builder->CreateBitCast(arg0, shifted_one->getType());
        tmp = builder->CreateXor(arg0, builder->CreateAnd(shifted_one, arg1));
        tmp = builder->CreateBitCast(tmp, common_llvm_type);
    }

    template <typename T>
    bool generate_optimization_instructions(const T* routine, ASR::call_arg_t* m_args) {
        std::string routine_name = std::string(routine->m_name);
        if( routine_name.find("flipsign") != std::string::npos ) {
            generate_flip_sign(m_args);
            return true;
        } else if( routine_name.find("fma") != std::string::npos ) {
            generate_fma(m_args);
            return true;
        } else if( routine_name.find("signfromvalue") != std::string::npos ) {
            generate_sign_from_value(m_args);
            return true;
        }
        return false;
    }

    int get_class_hash(ASR::symbol_t* class_sym) {
        if( type2vtabid.find(class_sym) == type2vtabid.end() ) {
            type2vtabid[class_sym] = type2vtabid.size();
        }
        return type2vtabid[class_sym];
    }

    llvm::Value* convert_to_polymorphic_arg(ASR::expr_t* arg_expr, llvm::Value* dt, ASR::expr_t* s_m_args0,
        ASR::ttype_t* s_m_args0_type, ASR::ttype_t* arg_type) {
        if( !ASRUtils::is_class_type(ASRUtils::type_get_past_array(s_m_args0_type)) ) {
            return dt;
        }
       
        if( ASRUtils::is_unlimited_polymorphic_type(s_m_args0) ) {
            if (ASRUtils::is_class_type(arg_type)) {
                if( ASRUtils::is_array(s_m_args0_type) ) {
                    // TODO: Handle this case
                    return dt;
                } else {
                    llvm::Type* _type = llvm_utils->get_type_from_ttype_t_util(s_m_args0, s_m_args0_type, module.get());
                    llvm::Type* dt_type = llvm_utils->get_type_from_ttype_t_util(arg_expr, arg_type, module.get());
                    ASR::ttype_t* wrapped_struct_type = ASRUtils::make_StructType_t_util(al, arg_expr->base.loc,
                                    ASRUtils::symbol_get_past_external(ASRUtils::get_struct_sym_from_struct_expr(arg_expr)));
                    llvm::Type* wrapped_struct_llvm_type = llvm_utils->get_type_from_ttype_t_util(arg_expr, wrapped_struct_type, module.get());
                    llvm::Value* abstract_ = llvm_utils->CreateAlloca(*builder, _type);
                    llvm::Value* polymorphic_addr = llvm_utils->create_gep2(_type, abstract_, 1);
                    builder->CreateStore(
                        builder->CreateBitCast(
                            llvm_utils->CreateLoad2(wrapped_struct_llvm_type->getPointerTo(),
                                                    llvm_utils->create_gep2(dt_type, dt, 1)),
                            llvm::Type::getVoidTy(context)->getPointerTo()),
                        polymorphic_addr);
                    llvm::Value* type_id_addr = llvm_utils->create_gep2(_type, abstract_, 0);
                    if (ASR::is_a<ASR::StructType_t>(*arg_type)) {
                        llvm::Value* hash = llvm_utils->create_gep2(dt_type, dt, 0);
                        hash = llvm_utils->CreateLoad2(llvm_utils->getIntType(8), hash);
                        builder->CreateStore(hash, type_id_addr);
                    } else {
                        // Case: when integer, real, character etc. passed to class(*) argument
                        llvm::Value* hash = llvm::ConstantInt::get(llvm_utils->getIntType(8),
                            llvm::APInt(64, -((int) arg_type->type) -
                            ASRUtils::extract_kind_from_ttype_t(arg_type), true));
                        builder->CreateStore(hash, type_id_addr);
                    }
                    return abstract_;
                }
            }
            if( ASRUtils::is_array(s_m_args0_type) ) {
                llvm::Type* array_type = llvm_utils->get_type_from_ttype_t_util(s_m_args0, s_m_args0_type, module.get());
                llvm::Value* abstract_array = llvm_utils->CreateAlloca(*builder, array_type);
                llvm::Type* array_data_type = llvm_utils->get_el_type(s_m_args0,
                    ASRUtils::type_get_past_array(s_m_args0_type), module.get());
                llvm::Type* dt_array_data_type = llvm_utils->get_el_type(arg_expr,
                    ASRUtils::type_get_past_array(arg_type), module.get());
                llvm::Value* array_data = llvm_utils->CreateAlloca(*builder, array_data_type);
                builder->CreateStore(array_data,
                    arr_descr->get_pointer_to_data(abstract_array));
                arr_descr->fill_array_details(arg_expr, s_m_args0, dt, abstract_array, arg_type, s_m_args0_type, module.get(), true);
                llvm::Value* polymorphic_data = llvm_utils->CreateLoad2(array_data_type->getPointerTo(),
                    arr_descr->get_pointer_to_data(abstract_array));
                llvm::Value* polymorphic_data_addr = llvm_utils->create_gep2(array_data_type, polymorphic_data, 1);
                llvm::Value* dt_data = llvm_utils->CreateLoad2(dt_array_data_type->getPointerTo(), arr_descr->get_pointer_to_data(dt));
                builder->CreateStore(
                    builder->CreateBitCast(dt_data, llvm::Type::getVoidTy(context)->getPointerTo()),
                    polymorphic_data_addr);
                llvm::Value* type_id_addr = llvm_utils->create_gep2(array_data_type, polymorphic_data, 0);
                builder->CreateStore(
                    llvm::ConstantInt::get(llvm_utils->getIntType(8),
                        llvm::APInt(64, -((int) ASRUtils::type_get_past_array(arg_type)->type) -
                            ASRUtils::extract_kind_from_ttype_t(arg_type), true)),
                    type_id_addr);
                return abstract_array;
            } else {
                llvm::Type* _type = llvm_utils->get_type_from_ttype_t_util(s_m_args0, s_m_args0_type, module.get());
                llvm::Value* abstract_ = llvm_utils->CreateAlloca(*builder, _type);
                llvm::Value* polymorphic_addr = llvm_utils->create_gep2(_type, abstract_, 1);
                builder->CreateStore(
                    builder->CreateBitCast(dt, llvm::Type::getVoidTy(context)->getPointerTo()),
                    polymorphic_addr);
                llvm::Value* type_id_addr = llvm_utils->create_gep2(_type, abstract_, 0);
                if (ASR::is_a<ASR::StructType_t>(*arg_type) && !ASRUtils::is_class_type(arg_type)) {
                    ASR::symbol_t* struct_sym = ASRUtils::symbol_get_past_external(ASRUtils::symbol_get_past_external(ASRUtils::get_struct_sym_from_struct_expr(arg_expr)));
                    llvm::Value* hash = llvm::ConstantInt::get(llvm_utils->getIntType(8),
                    llvm::APInt(64, get_class_hash(struct_sym)));
                    builder->CreateStore(hash, type_id_addr);
                } else {  
                    // Case: when integer, real, character etc. passed to class(*) argument
                    llvm::Value* hash = llvm::ConstantInt::get(llvm_utils->getIntType(8),
                        llvm::APInt(64, -((int) arg_type->type) -
                        ASRUtils::extract_kind_from_ttype_t(arg_type), true));
                    builder->CreateStore(hash, type_id_addr);
                }
                return abstract_;
            }
        } else if( ASR::is_a<ASR::StructType_t>(*ASRUtils::type_get_past_array(arg_type)) ) {
            ASR::StructType_t* struct_t = ASR::down_cast<ASR::StructType_t>(ASRUtils::type_get_past_array(arg_type));
            if (struct_t->m_is_cstruct) {
                ASR::symbol_t* struct_sym = ASRUtils::symbol_get_past_external(ASRUtils::symbol_get_past_external(ASRUtils::get_struct_sym_from_struct_expr(arg_expr)));
                if( type2vtab.find(struct_sym) == type2vtab.end() &&
                    type2vtab[struct_sym].find(current_scope) == type2vtab[struct_sym].end() ) {
                    create_vtab_for_struct_type(struct_sym, current_scope);
                }
                if (ASRUtils::is_array(s_m_args0_type) &&
                    ASRUtils::extract_physical_type(s_m_args0_type) == ASR::array_physical_typeType::DescriptorArray) {
                    // TODO: Handle convert of descriptor arrays
                }
                llvm::Value* dt_polymorphic = llvm_utils->CreateAlloca(*builder,
                    llvm_utils->getClassType(ASR::down_cast<ASR::Struct_t>(
                            ASRUtils::symbol_get_past_external(ASRUtils::get_struct_sym_from_struct_expr(s_m_args0))),
                            LLVM::is_llvm_pointer(*s_m_args0_type)));
                llvm::Type* _type = llvm_utils->get_type_from_ttype_t_util(s_m_args0,
                    ASRUtils::type_get_past_array(s_m_args0_type), module.get());

                llvm::Value* hash_ptr = llvm_utils->create_gep2(_type, dt_polymorphic, 0);
                llvm::Value* hash = llvm::ConstantInt::get(llvm_utils->getIntType(8), llvm::APInt(64, get_class_hash(struct_sym)));
                builder->CreateStore(hash, hash_ptr);
                llvm::Value* class_ptr = llvm_utils->create_gep2(_type, dt_polymorphic, 1);

                builder->CreateStore(builder->CreateBitCast(dt, llvm_utils->getStructType(ASR::down_cast<ASR::Struct_t>(
                        ASRUtils::symbol_get_past_external(ASRUtils::get_struct_sym_from_struct_expr(s_m_args0))), module.get(), true)), class_ptr);

                return dt_polymorphic;
            } else {
                // No need to convert if types are same
                if (llvm_utils->getClassType(
                        ASR::down_cast<ASR::Struct_t>(
                            ASRUtils::symbol_get_past_external(ASRUtils::get_struct_sym_from_struct_expr(s_m_args0))),
                        true)
                    == llvm_utils->getClassType(ASR::down_cast<ASR::Struct_t>(
                            ASRUtils::symbol_get_past_external(ASRUtils::get_struct_sym_from_struct_expr(arg_expr))), true)) {
                    return dt;
                }
                ASR::symbol_t* struct_sym = ASRUtils::symbol_get_past_external(ASRUtils::symbol_get_past_external(ASRUtils::get_struct_sym_from_struct_expr(arg_expr)));
                if( type2vtab.find(struct_sym) == type2vtab.end() &&
                    type2vtab[struct_sym].find(current_scope) == type2vtab[struct_sym].end() ) {
                    create_vtab_for_struct_type(struct_sym, current_scope);
                }

                llvm::Value* dt_polymorphic = llvm_utils->CreateAlloca(*builder,
                llvm_utils->getClassType(ASR::down_cast<ASR::Struct_t>(
                            ASRUtils::symbol_get_past_external(ASRUtils::get_struct_sym_from_struct_expr(s_m_args0))), LLVM::is_llvm_pointer(*s_m_args0_type)));
                llvm::Type* _type = llvm_utils->get_type_from_ttype_t_util(s_m_args0, s_m_args0_type, module.get());
                llvm::Value* hash_ptr = llvm_utils->create_gep2(_type, dt_polymorphic, 0);
                llvm::Value* hash = llvm::ConstantInt::get(llvm_utils->getIntType(8), llvm::APInt(64, get_class_hash(struct_sym)));
                builder->CreateStore(hash, hash_ptr);

                llvm::Type* dt_base_type = llvm_utils->get_type_from_ttype_t_util(arg_expr, arg_type, module.get());
                // Convert input arg from polymorphic to normal
                dt = llvm_utils->create_gep2(dt_base_type, dt, 1);
                dt = llvm_utils->CreateLoad2(dt_base_type->getPointerTo(),dt);

                ASR::symbol_t* dest_struct_sym = ASRUtils::symbol_get_past_external(ASRUtils::symbol_get_past_external(ASRUtils::get_struct_sym_from_struct_expr(s_m_args0)));
                llvm::Type* dest_ty = name2dertype[ASRUtils::symbol_name(dest_struct_sym)]->getPointerTo();
                dt = builder->CreateBitCast(dt, dest_ty);

                llvm::Value* class_ptr = llvm_utils->create_gep2(_type, dt_polymorphic, 1);
                builder->CreateStore(dt, class_ptr);

                return dt_polymorphic;
            }
        }
        return dt;
    }

    void visit_SubroutineCall(const ASR::SubroutineCall_t &x) {
        if (compiler_options.emit_debug_info) debug_emit_loc(x);
        if( ASRUtils::is_intrinsic_optimization(x.m_name) ) {
            ASR::Function_t* routine = ASR::down_cast<ASR::Function_t>(
                        ASRUtils::symbol_get_past_external(x.m_name));
            if( generate_optimization_instructions(routine, x.m_args) ) {
                return ;
            }
        }

        std::vector<llvm::Value*> args;
        if( x.m_dt && ASR::is_a<ASR::StructInstanceMember_t>(*x.m_dt) &&
            ASR::is_a<ASR::Variable_t>(*ASRUtils::symbol_get_past_external(x.m_name)) &&
            ASR::is_a<ASR::FunctionType_t>(*ASRUtils::symbol_type(x.m_name)) ) {
            uint64_t ptr_loads_copy = ptr_loads;
            ptr_loads = 1;
            this->visit_expr(*x.m_dt);
            ptr_loads = ptr_loads_copy;
            llvm::Value* callee = llvm_utils->CreateLoad(tmp);

            args = convert_call_args(x, false);
            ASR::Function_t* func = nullptr;
            if (ASR::is_a<ASR::Variable_t>(*ASRUtils::symbol_get_past_external(x.m_name))) {
                ASR::Variable_t* var = ASR::down_cast<ASR::Variable_t>(
                    ASRUtils::symbol_get_past_external(x.m_name));
                func = ASR::down_cast<ASR::Function_t>(
                    ASRUtils::symbol_get_past_external(var->m_type_declaration));
            } else if (ASR::is_a<ASR::Function_t>(*ASRUtils::symbol_get_past_external(x.m_name))) {
                func = ASR::down_cast<ASR::Function_t>(
                    ASRUtils::symbol_get_past_external(x.m_name));
            }
            llvm::FunctionType* fntype = llvm_utils->get_function_type(*func, module.get());
            tmp = builder->CreateCall(fntype, callee, args);
            return ;
        }

        const ASR::symbol_t *proc_sym = symbol_get_past_external(x.m_name);
        std::string proc_sym_name = "";
        bool is_runtime_polymorphism = false;
        bool is_nopass = false;
        if( ASR::is_a<ASR::StructMethodDeclaration_t>(*proc_sym) ) {
            ASR::StructMethodDeclaration_t* class_proc =
                ASR::down_cast<ASR::StructMethodDeclaration_t>(proc_sym);
            is_runtime_polymorphism = ASRUtils::is_class_type(ASRUtils::extract_type(ASRUtils::expr_type(x.m_dt)));
            proc_sym_name = class_proc->m_name;
            is_nopass = class_proc->m_is_nopass;
        }
        ASR::Function_t *s;
        char* self_argument = nullptr;
        llvm::Value* pass_arg = nullptr;
        if (ASR::is_a<ASR::Function_t>(*proc_sym)) {
            s = ASR::down_cast<ASR::Function_t>(proc_sym);
        } else if (ASR::is_a<ASR::StructMethodDeclaration_t>(*proc_sym)) {
            ASR::StructMethodDeclaration_t *clss_proc = ASR::down_cast<
                ASR::StructMethodDeclaration_t>(proc_sym);
            s = ASR::down_cast<ASR::Function_t>(clss_proc->m_proc);
            self_argument = clss_proc->m_self_argument;
            proc_sym = clss_proc->m_proc;
        } else if (ASR::is_a<ASR::Variable_t>(*proc_sym)) {
            ASR::symbol_t *type_decl = ASR::down_cast<ASR::Variable_t>(proc_sym)->m_type_declaration;
            LCOMPILERS_ASSERT(type_decl && ASR::is_a<ASR::Function_t>(*ASRUtils::symbol_get_past_external(type_decl)));
            s = ASR::down_cast<ASR::Function_t>(ASRUtils::symbol_get_past_external(type_decl));
        } else {
            throw CodeGenError("SubroutineCall: Symbol type not supported");
        }
        if( s == nullptr ) {
            s = ASR::down_cast<ASR::Function_t>(symbol_get_past_external(x.m_name));
        }
        bool is_method = false;
        if (x.m_dt && (!is_nopass)) {
            is_method = true;
            if (ASR::is_a<ASR::Var_t>(*x.m_dt)) {
                ASR::Variable_t *caller = EXPR2VAR(x.m_dt);
                std::uint32_t h = get_hash((ASR::asr_t*)caller);
                // declared variable in the current scope
                llvm::Value* dt = llvm_symtab[h];
                // Function class type
                ASR::ttype_t* s_m_args0_type = ASRUtils::type_get_past_pointer(
                                                ASRUtils::expr_type(s->m_args[0]));
                // derived type declared type
                ASR::ttype_t* dt_type = ASRUtils::type_get_past_allocatable(ASRUtils::type_get_past_pointer(caller->m_type));
                dt = convert_to_polymorphic_arg(x.m_dt, dt, s->m_args[0], s_m_args0_type, dt_type);
                args.push_back(dt);
            } else if (ASR::is_a<ASR::StructInstanceMember_t>(*x.m_dt)) {
                // Declared struct variable
                this->visit_expr(*x.m_dt);
                llvm::Value* dt = tmp;
                ASR::ttype_t* caller_type = ASRUtils::type_get_past_allocatable_pointer(ASRUtils::expr_type(x.m_dt));

                // Function's class type
                ASR::ttype_t *s_m_args0_type;
                ASR::expr_t* s_m_args0 = nullptr;
                if (self_argument != nullptr) {
                    ASR::symbol_t *class_sym = s->m_symtab->resolve_symbol(self_argument);
                    ASR::Variable_t *var = ASR::down_cast<ASR::Variable_t>(class_sym);
                    s_m_args0 = ASRUtils::EXPR(ASR::make_Var_t(al, var->base.base.loc, &var->base));
                    s_m_args0_type = ASRUtils::type_get_past_allocatable(
                        ASRUtils::type_get_past_pointer(var->m_type));
                    s_m_args0 = ASRUtils::EXPR(ASR::make_Var_t(al, var->base.base.loc, (ASR::symbol_t*) var));
                } else {
                    s_m_args0 = s->m_args[0];
                    s_m_args0_type = ASRUtils::type_get_past_allocatable(
                        ASRUtils::type_get_past_pointer(
                        ASRUtils::expr_type(s->m_args[0])));
                }

                // Convert to polymorphic argument
                llvm::Value* dt_polymorphic = convert_to_polymorphic_arg(x.m_dt, dt, s_m_args0, s_m_args0_type, caller_type);

                if (self_argument == nullptr) {
                    args.push_back(dt_polymorphic);
                } else {
                    pass_arg = dt_polymorphic;
                }
            } else if (ASR::is_a<ASR::ArrayItem_t>(*x.m_dt)){
                this->visit_expr(*x.m_dt);
                llvm::Value* dt = tmp;
                llvm::Value* dt_polymorphic = tmp;
                dt_polymorphic = convert_to_polymorphic_arg(x.m_dt, dt, s->m_args[0], ASRUtils::expr_type(s->m_args[0]),
                                                ASRUtils::expr_type(x.m_dt));
                args.push_back(dt_polymorphic);
            } else {
                throw CodeGenError("SubroutineCall: StructType symbol type not supported");
            }
        }
        if (is_runtime_polymorphism) {
            if (is_nopass || args.empty()) {
                visit_RuntimePolymorphicSubroutineCall(x, proc_sym_name, nullptr);
            } else {
                visit_RuntimePolymorphicSubroutineCall(x, proc_sym_name, args[0]);
            }
            return;
        }
        std::string sub_name = s->m_name;
        uint32_t h;
        ASR::FunctionType_t* s_func_type = ASR::down_cast<ASR::FunctionType_t>(s->m_function_signature);
        if (s_func_type->m_abi == ASR::abiType::LFortranModule) {
            throw CodeGenError("Subroutine LCompilers interfaces not implemented yet");
        } else if (s_func_type->m_abi == ASR::abiType::ExternalUndefined) {
            h = get_hash((ASR::asr_t*)proc_sym);
        } else if (s_func_type->m_abi == ASR::abiType::Source) {
            h = get_hash((ASR::asr_t*)proc_sym);
        } else if (s_func_type->m_abi == ASR::abiType::BindC) {
            h = get_hash((ASR::asr_t*)proc_sym);
        } else if (s_func_type->m_abi == ASR::abiType::Intrinsic) {
            if (sub_name == "get_command_argument") {
                llvm::Function *fn = module->getFunction("_lpython_get_argv");
                if (!fn) {
                    llvm::FunctionType *function_type = llvm::FunctionType::get(
                        character_type, {
                            llvm::Type::getInt32Ty(context)
                        }, false);
                    fn = llvm::Function::Create(function_type,
                        llvm::Function::ExternalLinkage, "_lpython_get_argv", *module);
                }
                args = convert_call_args(x, is_method);
                LCOMPILERS_ASSERT(args.size() > 0);
                tmp = builder->CreateCall(fn, {llvm_utils->CreateLoad2(
                    llvm::Type::getInt32Ty(context), args[0])});
                if (args.size() > 1)
                    builder->CreateStore(tmp, args[1]);
                return;
            } else if (sub_name == "get_environment_variable") {
                llvm::Function *fn = module->getFunction("_lfortran_get_env_variable");
                if (!fn) {
                    llvm::FunctionType *function_type = llvm::FunctionType::get(
                        character_type, {
                            character_type
                        }, false);
                    fn = llvm::Function::Create(function_type,
                        llvm::Function::ExternalLinkage, "_lfortran_get_env_variable", *module);
                }
                args = convert_call_args(x, is_method);
                LCOMPILERS_ASSERT(args.size() > 0);
                tmp = builder->CreateCall(fn, {llvm_utils->CreateLoad(args[0])});
                if (args.size() > 1)
                    builder->CreateStore(tmp, args[1]);
                return;
            } else if (sub_name == "_lcompilers_execute_command_line_") {
                llvm::Function *fn = module->getFunction("_lfortran_exec_command");
                if (!fn) {
                    llvm::FunctionType *function_type = llvm::FunctionType::get(
                        llvm::Type::getInt32Ty(context), {
                            character_type
                        }, false);
                    fn = llvm::Function::Create(function_type,
                        llvm::Function::ExternalLinkage, "_lfortran_exec_command", *module);
                }
                args = convert_call_args(x, is_method);
                LCOMPILERS_ASSERT(args.size() > 0);
                tmp = builder->CreateCall(fn, {llvm_utils->CreateLoad(args[0])});
                return;
            }
            h = get_hash((ASR::asr_t*)proc_sym);
        } else {
            throw CodeGenError("ABI type not implemented yet in SubroutineCall.");
        }

        if (llvm_symtab_fn_arg.find(h) != llvm_symtab_fn_arg.end()) {
            // Check if this is a callback function
            llvm::Value* fn = llvm_symtab_fn_arg[h];
            llvm::FunctionType* fntype = llvm_symtab_fn[h]->getFunctionType();
            std::string m_name = ASRUtils::symbol_name(x.m_name);
            if ( x.m_original_name && ASR::is_a<ASR::Variable_t>(*x.m_original_name) ) {
                ASR::Variable_t* x_m_original_name = ASR::down_cast<ASR::Variable_t>(x.m_original_name);
                if ( x_m_original_name->m_intent == ASRUtils::intent_out || x_m_original_name->m_intent == ASRUtils::intent_inout ) {
                    fn = llvm_utils->CreateLoad2(llvm_utils->get_type_from_ttype_t_util(ASRUtils::EXPR(ASR::make_Var_t(
                    al, x_m_original_name->base.base.loc, &x_m_original_name->base)), x_m_original_name->m_type, module.get()), fn);
                }
            }
            args = convert_call_args(x, is_method);
            tmp = builder->CreateCall(fntype, fn, args);
        } else if (ASR::is_a<ASR::Variable_t>(*proc_sym) &&
                llvm_symtab.find(h) != llvm_symtab.end()) {
            llvm::Value* fn = llvm_symtab[h];
            fn = llvm_utils->CreateLoad(fn);
            ASR::Variable_t* v = ASR::down_cast<ASR::Variable_t>(proc_sym);
            llvm::FunctionType* fntype = llvm_utils->get_function_type(*ASR::down_cast<ASR::Function_t>(ASRUtils::symbol_get_past_external(v->m_type_declaration)), module.get());
            std::string m_name = ASRUtils::symbol_name(x.m_name);
            args = convert_call_args(x, is_method);
            tmp = builder->CreateCall(fntype, fn, args);
        } else if (llvm_symtab_fn.find(h) == llvm_symtab_fn.end()) {
            throw CodeGenError("Subroutine code not generated for '"
                + std::string(s->m_name) + "'");
            return;
        } else {
            llvm::Function *fn = llvm_symtab_fn[h];
            std::string m_name = ASRUtils::symbol_name(x.m_name);
            std::vector<llvm::Value *> args2 = convert_call_args(x, is_method);
            args.insert(args.end(), args2.begin(), args2.end());
            // check if type of each arg is same as type of each arg in subrout_called
            if (ASR::is_a<ASR::Function_t>(*symbol_get_past_external(x.m_name))) {
                ASR::Function_t* subrout_called = ASR::down_cast<ASR::Function_t>(symbol_get_past_external(x.m_name));
                for (size_t i = 0; i < subrout_called->n_args; i++) {
                    ASR::expr_t* expected_arg = subrout_called->m_args[i];
                    ASR::expr_t* passed_arg = x.m_args[i].m_value;
                    ASR::ttype_t* expected_arg_type = ASRUtils::expr_type(expected_arg);
                    ASR::ttype_t* passed_arg_type = ASRUtils::expr_type(passed_arg);
                    if (ASR::is_a<ASR::ArrayItem_t>(*passed_arg)) {
                        if (!ASRUtils::types_equal(expected_arg_type, passed_arg_type, true)) {
                            throw CodeGenError("Type mismatch in subroutine call, expected `" + ASRUtils::type_to_str_python(expected_arg_type)
                                    + "`, passed `" + ASRUtils::type_to_str_python(passed_arg_type) + "`", x.m_args[i].m_value->base.loc);
                        }
                    }
                }
            }
            if (pass_arg) {
                args.push_back(pass_arg);
            }
            builder->CreateCall(fn, args);
        }
    }

    void handle_bitwise_args(const ASR::FunctionCall_t& x, llvm::Value*& arg1,
                             llvm::Value*& arg2) {
        LCOMPILERS_ASSERT(x.n_args == 2);
        tmp = nullptr;
        this->visit_expr_wrapper(x.m_args[0].m_value, true);
        arg1 = tmp;
        tmp = nullptr;
        this->visit_expr_wrapper(x.m_args[1].m_value, true);
        arg2 = tmp;
    }

    void handle_bitwise_xor(const ASR::FunctionCall_t& x) {
        llvm::Value *arg1 = nullptr, *arg2 = nullptr;
        handle_bitwise_args(x, arg1, arg2);
        tmp = builder->CreateXor(arg1, arg2);
    }

    void handle_bitwise_and(const ASR::FunctionCall_t& x) {
        llvm::Value *arg1 = nullptr, *arg2 = nullptr;
        handle_bitwise_args(x, arg1, arg2);
        tmp = builder->CreateAnd(arg1, arg2);
    }

    void handle_bitwise_or(const ASR::FunctionCall_t& x) {
        llvm::Value *arg1 = nullptr, *arg2 = nullptr;
        handle_bitwise_args(x, arg1, arg2);
        tmp = builder->CreateOr(arg1, arg2);
    }

    void handle_allocated(ASR::expr_t* arg) {
        ASR::ttype_t* asr_type = ASRUtils::expr_type(arg);
        int64_t ptr_loads_copy = ptr_loads;
        if (ASRUtils::is_class_type(ASRUtils::extract_type(asr_type))) {
            ptr_loads = 0;
            visit_expr_wrapper(arg, false);
        } else {
            ptr_loads = 1;
            visit_expr_wrapper(arg, true);
        }
        ptr_loads = ptr_loads_copy;
        int n_dims = ASRUtils::extract_n_dims_from_ttype(asr_type);
        if (ASRUtils::is_class_type(ASRUtils::extract_type(asr_type))) {
            // If the pointer is class, get its type pointer
            llvm::Type* class_type = llvm_utils->get_type_from_ttype_t_util(arg, asr_type, module.get());
            tmp = llvm_utils->create_gep2(class_type, tmp, 1);
            tmp = llvm_utils->CreateLoad2(class_type->getPointerTo(), tmp);
        }
        if( n_dims > 0 ) {
            visit_expr_load_wrapper(arg, 1, true);
#if LLVM_VERSION_MAJOR > 16
            llvm::Type* arr_type = llvm_utils->get_type_from_ttype_t_util(arg,
                ASRUtils::type_get_past_allocatable_pointer(asr_type),
                module.get(), ASRUtils::expr_abi(arg));
            ptr_type[tmp] = arr_type;
#endif
            llvm::Type* llvm_data_type = llvm_utils->get_type_from_ttype_t_util(arg,
                ASRUtils::extract_type(asr_type), module.get(), ASRUtils::expr_abi(arg));
            tmp = arr_descr->get_is_allocated_flag(tmp, llvm_data_type, arg);
        } else if (ASRUtils::is_character(*expr_type(arg))) {
            visit_expr_load_wrapper(arg, 0);
            tmp = builder->CreateICmpNE(
                llvm_utils->get_string_data(
                    ASR::down_cast<ASR::String_t>(ASRUtils::extract_type(expr_type(arg))),
                    tmp),
                llvm::ConstantPointerNull::get(
                    character_type));
        } else {
            tmp = builder->CreateICmpNE(
                builder->CreatePtrToInt(tmp, llvm::Type::getInt64Ty(context)),
                    llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), llvm::APInt(64, 0)) );
        }
    }

    llvm::Value* CreatePointerToStructTypeReturnValue(llvm::FunctionType* fnty,
                                                  llvm::Value* return_value,
                                                  ASR::ttype_t* asr_return_type) {
        if( !LLVM::is_llvm_struct(asr_return_type) ) {
            return return_value;
        }

        // Call to LLVM APIs not needed to fetch the return type of the function.
        // We can use asr_return_type as well but anyways for compactness I did it here.
        llvm::Value* pointer_to_struct = llvm_utils->CreateAlloca(*builder, fnty->getReturnType());
        LLVM::CreateStore(*builder, return_value, pointer_to_struct);
        return pointer_to_struct;
    }

    llvm::Value* CreateCallUtil(llvm::FunctionType* fnty, llvm::Function* fn,
                                std::vector<llvm::Value*>& args,
                                ASR::ttype_t* asr_return_type) {
        llvm::Value* return_value = builder->CreateCall(fn, args);
        return CreatePointerToStructTypeReturnValue(fnty, return_value,
                                                asr_return_type);
    }

    llvm::Value* CreateCallUtil(llvm::Function* fn, std::vector<llvm::Value*>& args,
                                ASR::ttype_t* asr_return_type) {
        return CreateCallUtil(fn->getFunctionType(), fn, args, asr_return_type);
    }

    void visit_RuntimePolymorphicSubroutineCall(const ASR::SubroutineCall_t& x, std::string proc_sym_name, llvm::Value *llvm_polymorphic) {
        std::vector<std::pair<llvm::Value*, ASR::symbol_t*>> vtabs;
        ASR::Struct_t* dt_sym_type = nullptr;
        ASR::ttype_t* dt_ttype_t = ASRUtils::type_get_past_allocatable(ASRUtils::type_get_past_pointer(
                                        ASRUtils::expr_type(x.m_dt)));
        if( ASR::is_a<ASR::StructType_t>(*dt_ttype_t) ) {
            dt_sym_type = ASR::down_cast<ASR::Struct_t>(
                ASRUtils::symbol_get_past_external(ASRUtils::symbol_get_past_external(ASRUtils::get_struct_sym_from_struct_expr(x.m_dt))));
            create_vtab_for_struct_type(&dt_sym_type->base, current_scope);
        }
        LCOMPILERS_ASSERT(dt_sym_type != nullptr);
        for( auto& item: type2vtab ) {
            ASR::Struct_t* a_dt = ASR::down_cast<ASR::Struct_t>(item.first);
            if( !a_dt->m_is_abstract &&
                (a_dt == dt_sym_type ||
                ASRUtils::is_parent(a_dt, dt_sym_type) ||
                ASRUtils::is_parent(dt_sym_type, a_dt)) ) {
                for( auto& item2: item.second ) {
                    if( item2.first == current_scope ) {
                        // Find the Struct symbol to which the StructMethodDeclaration belongs
                        bool found = false;
                        while (a_dt) {
                            for (auto &member : a_dt->m_symtab->get_scope()) {
                                if (ASR::is_a<ASR::StructMethodDeclaration_t>(*member.second)) {
                                    ASR::StructMethodDeclaration_t *proc_s = ASR::down_cast<ASR::StructMethodDeclaration_t>(member.second);
                                    if (proc_s->m_name == proc_sym_name) {
                                        found = true;
                                        break;
                                    }
                                }
                            }

                            if (found)
                                break;

                            if (a_dt->m_parent != nullptr) {
                                a_dt = ASR::down_cast<ASR::Struct_t>(ASRUtils::symbol_get_past_external(a_dt->m_parent));
                            } else {
                                a_dt = nullptr;
                            }
                        }
                        if (a_dt) vtabs.push_back(std::make_pair(item2.second, (ASR::symbol_t *)a_dt));
                    }
                }
            }
        }

        llvm::Value *llvm_dt = nullptr;
        if (llvm_polymorphic == nullptr) {
            uint64_t ptr_loads_copy = ptr_loads;
            ptr_loads = 0;
            this->visit_expr_wrapper(x.m_dt);
            ptr_loads = ptr_loads_copy;
            llvm_dt = tmp;
        } else {
            llvm_dt = llvm_polymorphic;
        }
        llvm::BasicBlock *mergeBB = llvm::BasicBlock::Create(context, "ifcont");
        llvm::Type* i64 = llvm::Type::getInt64Ty(context);
        llvm::Type* llvm_dt_type = nullptr;
        llvm_dt_type = llvm_utils->get_type_from_ttype_t_util(x.m_dt,
            ASRUtils::extract_type(ASRUtils::expr_type(x.m_dt)), module.get());
        llvm_dt = builder->CreateBitCast(llvm_dt, llvm_dt_type->getPointerTo());
        for( size_t i = 0; i < vtabs.size(); i++ ) {
            llvm::Function *fn = builder->GetInsertBlock()->getParent();

            llvm::BasicBlock *thenBB = llvm::BasicBlock::Create(context, "then", fn);
            llvm::BasicBlock *elseBB = llvm::BasicBlock::Create(context, "else");

            llvm::Value* vptr_int_hash = llvm_utils->CreateLoad2(i64, llvm_utils->create_gep2(llvm_dt_type, llvm_dt, 0));
            llvm::Type *dt_type = llvm_utils->getStructType(ASR::down_cast<ASR::Struct_t>(
                        ASRUtils::symbol_get_past_external(ASRUtils::get_struct_sym_from_struct_expr(x.m_dt))), module.get(), true);
            llvm::Value* dt_data = llvm_utils->CreateLoad2(dt_type, llvm_utils->create_gep2(llvm_dt_type, llvm_dt, 1));
            ASR::ttype_t* selector_var_type = ASRUtils::expr_type(x.m_dt);
            if( ASRUtils::is_array(selector_var_type) ) {
                vptr_int_hash = llvm_utils->CreateLoad2(vptr_int_hash->getType()->getPointerTo(), llvm_utils->create_gep2(vptr_int_hash->getType(), vptr_int_hash, 0));
            }
            ASR::symbol_t* type_sym = ASRUtils::symbol_get_past_external(vtabs[i].second);
            llvm::Value* type_sym_vtab = vtabs[i].first;
            llvm::Type* struct_ty = llvm::StructType::get(context, { i64 }, true);
            llvm::Value* vtab_obj_casted = builder->CreateBitCast(type_sym_vtab, struct_ty->getPointerTo());
            llvm::Value* cond = builder->CreateICmpEQ(
                                    vptr_int_hash,
                                    llvm_utils->CreateLoad2(i64,
                                        llvm_utils->create_gep2(struct_ty, vtab_obj_casted, 0) ) );

            builder->CreateCondBr(cond, thenBB, elseBB);
            builder->SetInsertPoint(thenBB);
            {
                std::vector<llvm::Value*> args;
                ASR::Struct_t* struct_type_t = ASR::down_cast<ASR::Struct_t>(type_sym);
                
                ASR::symbol_t* s_class_proc = struct_type_t->m_symtab->resolve_symbol(proc_sym_name);
                ASR::StructMethodDeclaration_t* class_proc = ASR::down_cast<ASR::StructMethodDeclaration_t>(s_class_proc);
                if (!class_proc->m_is_nopass) {
                    // add the self argument only when the class procedure has the `pass` attribute
                    llvm::Type* target_dt_type = llvm_utils->getStructType(struct_type_t, module.get(), true);
                    llvm::Type* target_class_dt_type = llvm_utils->getClassType(struct_type_t);
                    llvm::Value* target_dt = llvm_utils->CreateAlloca(*builder, target_class_dt_type);
                    llvm::Value* target_dt_hash_ptr = llvm_utils->create_gep2(target_class_dt_type, target_dt, 0);
                    builder->CreateStore(vptr_int_hash, target_dt_hash_ptr);
                    llvm::Value* target_dt_data_ptr = llvm_utils->create_gep2(target_class_dt_type, target_dt, 1);
                    builder->CreateStore(builder->CreateBitCast(dt_data, target_dt_type),
                                        target_dt_data_ptr);
                    args.push_back(target_dt);
                    std::string captured_global_name
                        = "__lcompilers_created__nested_context__" + proc_sym_name + "_self";
                    llvm::GlobalVariable* nested_global
                        = module->getNamedGlobal(captured_global_name);
                    if (nested_global) {
                        llvm::Type* nested_global_type = nested_global->getValueType();
                        llvm::Value* global_hash_ptr = llvm_utils->create_gep2(nested_global_type, nested_global, 0);
                        llvm::Value* global_data_ptr = llvm_utils->create_gep2(nested_global_type, nested_global, 1);
                        // Use the correct types for loading
                        llvm::Type* target_dt_hash_type = llvm::Type::getInt64Ty(context);
                        llvm::Type* target_dt_data_type = llvm_utils->getStructType(struct_type_t, module.get(), true);
                        llvm::Value* local_hash_val = llvm_utils->CreateLoad2(target_dt_hash_type,target_dt_hash_ptr);
                        llvm::Value* local_data_val = llvm_utils->CreateLoad2(target_dt_data_type, target_dt_data_ptr);

                        // Store the loaded values
                        builder->CreateStore(local_hash_val, global_hash_ptr);
                        builder->CreateStore(local_data_val, global_data_ptr);
                    }
                }
                ASR::symbol_t* s_proc = ASRUtils::symbol_get_past_external(class_proc->m_proc);
                uint32_t h = get_hash((ASR::asr_t*) s_proc);
                llvm::Function* fn = llvm_symtab_fn[h];
                std::vector<llvm::Value *> args2 = convert_call_args(x, !class_proc->m_is_nopass);
                args.insert(args.end(), args2.begin(), args2.end());
                builder->CreateCall(fn, args);
            }
            builder->CreateBr(mergeBB);

            start_new_block(elseBB);
            current_select_type_block_type = nullptr;
            current_select_type_block_type_asr = nullptr;
            current_select_type_block_der_type.clear();
        }
        start_new_block(mergeBB);
    }

    void visit_RuntimePolymorphicFunctionCall(const ASR::FunctionCall_t& x, std::string proc_sym_name) {
        std::vector<std::pair<llvm::Value*, ASR::symbol_t*>> vtabs;
        ASR::Struct_t* dt_sym_type = nullptr;
        ASR::ttype_t* dt_ttype_t = ASRUtils::type_get_past_allocatable(ASRUtils::type_get_past_pointer(
                                        ASRUtils::expr_type(x.m_dt)));
        if( ASR::is_a<ASR::StructType_t>(*dt_ttype_t) ) {
            dt_sym_type = ASR::down_cast<ASR::Struct_t>(ASRUtils::symbol_get_past_external(ASRUtils::get_struct_sym_from_struct_expr(x.m_dt)));
        }
        LCOMPILERS_ASSERT(dt_sym_type != nullptr);
        for( auto& item: type2vtab ) {
            ASR::Struct_t* a_dt = ASR::down_cast<ASR::Struct_t>(item.first);
            if( !a_dt->m_is_abstract &&
                (a_dt == dt_sym_type ||
                ASRUtils::is_parent(a_dt, dt_sym_type) ||
                ASRUtils::is_parent(dt_sym_type, a_dt)) ) {
                for( auto& item2: item.second ) {
                    if( item2.first == current_scope ) {
                        vtabs.push_back(std::make_pair(item2.second, item.first));
                    }
                }
            }
        }

        uint64_t ptr_loads_copy = ptr_loads;
        ptr_loads = 0;
        this->visit_expr_wrapper(x.m_dt);
        ptr_loads = ptr_loads_copy;
        llvm::Value* llvm_dt = tmp;
        tmp = llvm_utils->CreateAlloca(*builder, llvm_utils->get_type_from_ttype_t_util(const_cast<ASR::expr_t*>(&x.base), x.m_type, module.get()));
        llvm::BasicBlock *mergeBB = llvm::BasicBlock::Create(context, "ifcont");
        llvm::Type* i64 = llvm::Type::getInt64Ty(context);
        llvm::Type* llvm_dt_type = nullptr;
        llvm_dt_type = llvm_utils->get_type_from_ttype_t_util(x.m_dt,
            ASRUtils::extract_type(ASRUtils::expr_type(x.m_dt)), module.get());
        llvm_dt = builder->CreateBitCast(llvm_dt, llvm_dt_type->getPointerTo());
        for( size_t i = 0; i < vtabs.size(); i++ ) {
            llvm::Function *fn = builder->GetInsertBlock()->getParent();

            llvm::BasicBlock *thenBB = llvm::BasicBlock::Create(context, "then", fn);
            llvm::BasicBlock *elseBB = llvm::BasicBlock::Create(context, "else");

            llvm::Value* vptr_int_hash = llvm_utils->CreateLoad2(i64, llvm_utils->create_gep2(llvm_dt_type, llvm_dt, 0));
            llvm::Type *dt_type = llvm_utils->getStructType(ASR::down_cast<ASR::Struct_t>(
                        ASRUtils::symbol_get_past_external(ASRUtils::get_struct_sym_from_struct_expr(x.m_dt))), module.get(), true);
            llvm::Value* dt_data = llvm_utils->CreateLoad2(dt_type, llvm_utils->create_gep2(llvm_dt_type, llvm_dt, 1));
            ASR::ttype_t* selector_var_type = ASRUtils::expr_type(x.m_dt);
            if( ASRUtils::is_array(selector_var_type) ) {
                vptr_int_hash = llvm_utils->CreateLoad2(vptr_int_hash->getType()->getPointerTo(), llvm_utils->create_gep2(vptr_int_hash->getType(), vptr_int_hash, 0));
            }
            ASR::symbol_t* type_sym = ASRUtils::symbol_get_past_external(vtabs[i].second);
            llvm::Value* type_sym_vtab = vtabs[i].first;
            llvm::Type* struct_ty = llvm::StructType::get(context, { i64 }, true);
            llvm::Value* vtab_obj_casted = builder->CreateBitCast(type_sym_vtab, struct_ty->getPointerTo());
            llvm::Value* cond = builder->CreateICmpEQ(
                                    vptr_int_hash,
                                    llvm_utils->CreateLoad2(i64,
                                        llvm_utils->create_gep2(struct_ty, vtab_obj_casted, 0) ) );

            builder->CreateCondBr(cond, thenBB, elseBB);
            builder->SetInsertPoint(thenBB);
            {
                std::vector<llvm::Value*> args;
                ASR::Struct_t* struct_type_t = ASR::down_cast<ASR::Struct_t>(type_sym);
                ASR::symbol_t* s_class_proc = struct_type_t->m_symtab->resolve_symbol(proc_sym_name);
                ASR::StructMethodDeclaration_t* class_proc = ASR::down_cast<ASR::StructMethodDeclaration_t>(s_class_proc);
                if (!class_proc->m_is_nopass) {
                    // add the self argument only when the class procedure has the `pass` attribute
                    llvm::Type* target_dt_type = llvm_utils->getStructType(struct_type_t, module.get(), true);
                    llvm::Type* target_class_dt_type = llvm_utils->getClassType(struct_type_t);
                    llvm::Value* target_dt = llvm_utils->CreateAlloca(*builder, target_class_dt_type);
                    llvm::Value* target_dt_hash_ptr = llvm_utils->create_gep2(target_class_dt_type, target_dt, 0);
                    builder->CreateStore(vptr_int_hash, target_dt_hash_ptr);
                    llvm::Value* target_dt_data_ptr = llvm_utils->create_gep2(target_class_dt_type, target_dt, 1);
                    builder->CreateStore(builder->CreateBitCast(dt_data, target_dt_type),
                                        target_dt_data_ptr);
                    args.push_back(target_dt);
                }
                ASR::symbol_t* s_proc = ASRUtils::symbol_get_past_external(class_proc->m_proc);
                uint32_t h = get_hash((ASR::asr_t*) s_proc);
                llvm::Function* fn = llvm_symtab_fn[h];
                ASR::Function_t* s = ASR::down_cast<ASR::Function_t>(s_proc);
                LCOMPILERS_ASSERT(s != nullptr);
                std::vector<llvm::Value *> args2 = convert_call_args(x, !class_proc->m_is_nopass);
                args.insert(args.end(), args2.begin(), args2.end());
                ASR::ttype_t *return_var_type0 = EXPR2VAR(s->m_return_var)->m_type;
                builder->CreateStore(CreateCallUtil(fn, args, return_var_type0), tmp);
            }
            builder->CreateBr(mergeBB);

            start_new_block(elseBB);
            current_select_type_block_type = nullptr;
            current_select_type_block_der_type.clear();
        }
        start_new_block(mergeBB);
        tmp = llvm_utils->CreateLoad(tmp);
    }

    void visit_FunctionCall(const ASR::FunctionCall_t &x) {
        if ( compiler_options.emit_debug_info ) debug_emit_loc(x);
        if( ASRUtils::is_intrinsic_optimization(x.m_name) ) {
            ASR::Function_t* routine = ASR::down_cast<ASR::Function_t>(
                        ASRUtils::symbol_get_past_external(x.m_name));
            if( generate_optimization_instructions(routine, x.m_args) ) {
                return ;
            }
        }
        if (x.m_value) {
            this->visit_expr_wrapper(x.m_value, true);
            return ;
        }

        std::vector<llvm::Value*> args;
        if( x.m_dt && ASR::is_a<ASR::StructInstanceMember_t>(*x.m_dt) &&
            ASR::is_a<ASR::Variable_t>(*ASRUtils::symbol_get_past_external(x.m_name)) &&
            ASR::is_a<ASR::FunctionType_t>(*ASRUtils::symbol_type(x.m_name)) ) {
            uint64_t ptr_loads_copy = ptr_loads;
            ptr_loads = 1;
            this->visit_expr(*x.m_dt);
            ptr_loads = ptr_loads_copy;
            llvm::Value* callee = llvm_utils->CreateLoad(tmp);

            args = convert_call_args(x, false);
            llvm::FunctionType* fntype = llvm_utils->get_function_type(
                *ASR::down_cast<ASR::Function_t>(ASRUtils::symbol_get_past_external(x.m_name)),
                module.get());
            tmp = builder->CreateCall(fntype, callee, args);
            return ;
        }

        const ASR::symbol_t *proc_sym = symbol_get_past_external(x.m_name);
        std::string proc_sym_name = "";
        bool is_deferred = false;
        bool is_nopass = false;
        if( ASR::is_a<ASR::StructMethodDeclaration_t>(*proc_sym) ) {
            ASR::StructMethodDeclaration_t* class_proc =
                ASR::down_cast<ASR::StructMethodDeclaration_t>(proc_sym);
            is_deferred = class_proc->m_is_deferred;
            proc_sym_name = class_proc->m_name;
            is_nopass = class_proc->m_is_nopass;
        }
        if( is_deferred ) {
            visit_RuntimePolymorphicFunctionCall(x, proc_sym_name);
            return ;
        }

        ASR::Function_t *s = nullptr;
        std::string self_argument = "";
        if (ASR::is_a<ASR::Function_t>(*proc_sym)) {
            s = ASR::down_cast<ASR::Function_t>(proc_sym);
        } else if (ASR::is_a<ASR::StructMethodDeclaration_t>(*proc_sym)) {
            ASR::StructMethodDeclaration_t *clss_proc = ASR::down_cast<
                ASR::StructMethodDeclaration_t>(proc_sym);
            s = ASR::down_cast<ASR::Function_t>(clss_proc->m_proc);
            if (clss_proc->m_self_argument) {
                self_argument = std::string(clss_proc->m_self_argument);
            }
            proc_sym = clss_proc->m_proc;
        } else if (ASR::is_a<ASR::Variable_t>(*proc_sym)) {
            ASR::symbol_t *type_decl = ASR::down_cast<ASR::Variable_t>(proc_sym)->m_type_declaration;
            LCOMPILERS_ASSERT(type_decl);
            s = ASR::down_cast<ASR::Function_t>(type_decl);
        } else {
            throw CodeGenError("FunctionCall: Symbol type not supported");
        }
        if( s == nullptr ) {
            s = ASR::down_cast<ASR::Function_t>(symbol_get_past_external(x.m_name));
        }
        bool is_method = false;
        llvm::Value* pass_arg = nullptr;
        if (x.m_dt && (!is_nopass)) {
            is_method = true;
            if (ASR::is_a<ASR::Var_t>(*x.m_dt)) {
                ASR::Variable_t *caller = EXPR2VAR(x.m_dt);
                std::uint32_t h = get_hash((ASR::asr_t*)caller);
                // declared variable in the current scope
                llvm::Value* dt = llvm_symtab[h];
                // Function class type
                ASR::ttype_t* s_m_args0_type = ASRUtils::type_get_past_pointer(
                                                ASRUtils::expr_type(s->m_args[0]));
                // derived type declared type
                ASR::ttype_t* dt_type = ASRUtils::type_get_past_allocatable_pointer(caller->m_type);
                dt = convert_to_polymorphic_arg(x.m_dt, dt, s->m_args[0], s_m_args0_type, dt_type);
                args.push_back(dt);
            } else if (ASR::is_a<ASR::StructInstanceMember_t>(*x.m_dt)) {
                // Declared struct variable
                this->visit_expr(*x.m_dt);
                llvm::Value* dt = tmp;
                ASR::ttype_t* caller_type = ASRUtils::type_get_past_allocatable_pointer(ASRUtils::expr_type(x.m_dt));

                // Function's class type
                ASR::ttype_t *s_m_args0_type;
                ASR::expr_t *s_m_args0 = nullptr;
                if (self_argument.length() > 0) {
                    ASR::symbol_t *class_sym = s->m_symtab->resolve_symbol(self_argument);
                    ASR::Variable_t *var = ASR::down_cast<ASR::Variable_t>(class_sym);
                    s_m_args0 = ASRUtils::EXPR(ASR::make_Var_t(al, var->base.base.loc, &var->base));
                    s_m_args0_type = ASRUtils::type_get_past_allocatable(
                        ASRUtils::type_get_past_pointer(var->m_type));
                    s_m_args0 = ASRUtils::EXPR(ASR::make_Var_t(al, var->base.base.loc, (ASR::symbol_t*) var));
                } else {
                    s_m_args0 = s->m_args[0];
                    s_m_args0_type = ASRUtils::type_get_past_allocatable(
                        ASRUtils::type_get_past_pointer(
                        ASRUtils::expr_type(s->m_args[0])));
                }

                // Convert to polymorphic argument
                llvm::Value* dt_polymorphic = convert_to_polymorphic_arg(x.m_dt, dt, s_m_args0, s_m_args0_type, caller_type);

                if (self_argument.length() == 0) {
                    args.push_back(dt_polymorphic);
                } else {
                    pass_arg = dt_polymorphic;
                }
            } else if(ASR::is_a<ASR::ArrayItem_t>(*x.m_dt)) {
                this->visit_expr(*x.m_dt);
                llvm::Value* dt = tmp;
                llvm::Value* dt_polymorphic = tmp;
                dt_polymorphic = convert_to_polymorphic_arg(x.m_dt, dt, s->m_args[0], ASRUtils::expr_type(s->m_args[0]),
                                                ASRUtils::expr_type(x.m_dt));
                args.push_back(dt_polymorphic);
            } else {
                throw CodeGenError("FunctionCall: StructType symbol type not supported");
            }
        }
        if( ASRUtils::is_intrinsic_function2(s) ) {
            std::string symbol_name = ASRUtils::symbol_name(x.m_name);
            if( startswith(symbol_name, "_bitwise_xor") ) {
                handle_bitwise_xor(x);
                return ;
            }
            if( startswith(symbol_name, "_bitwise_and") ) {
                handle_bitwise_and(x);
                return ;
            }
            if( startswith(symbol_name, "_bitwise_or") ) {
                handle_bitwise_or(x);
                return ;
            }
        }

        bool intrinsic_function = ASRUtils::is_intrinsic_function2(s);
        uint32_t h;
        ASR::FunctionType_t* s_func_type = ASR::down_cast<ASR::FunctionType_t>(s->m_function_signature);
        if (s_func_type->m_abi == ASR::abiType::Source && !intrinsic_function) {
            h = get_hash((ASR::asr_t*)proc_sym);
        } else if (s_func_type->m_abi == ASR::abiType::LFortranModule) {
            throw CodeGenError("Function LCompilers interfaces not implemented yet");
        } else if (s_func_type->m_abi == ASR::abiType::ExternalUndefined) {
            h = get_hash((ASR::asr_t*)proc_sym);
        } else if (s_func_type->m_abi == ASR::abiType::BindC) {
            h = get_hash((ASR::asr_t*)proc_sym);
        } else if (s_func_type->m_abi == ASR::abiType::Intrinsic || intrinsic_function) {
            std::string func_name = s->m_name;
            if( fname2arg_type.find(func_name) != fname2arg_type.end() ) {
                h = get_hash((ASR::asr_t*)proc_sym);
            } else {
                if (func_name == "len") {
                    args = convert_call_args(x, is_method);
                    LCOMPILERS_ASSERT(args.size() == 3)
                    tmp = lfortran_str_len(args[0]);
                    return;
                } else if (func_name == "command_argument_count") {
                    llvm::Function *fn = module->getFunction("_lfortran_get_argc");
                    if(!fn) {
                        llvm::FunctionType *function_type = llvm::FunctionType::get(
                            llvm::Type::getInt32Ty(context), {}, false);
                        fn = llvm::Function::Create(function_type,
                            llvm::Function::ExternalLinkage, "_lfortran_get_argc", *module);
                    }
                    tmp = builder->CreateCall(fn, {});
                    return;
                } else if (func_name == "achar") {
                    // TODO: make achar just StringChr
                    this->visit_expr_wrapper(x.m_args[0].m_value, true);
                    tmp = lfortran_str_chr(tmp);
                    return;
                }
                if( ASRUtils::get_FunctionType(s)->m_deftype == ASR::deftypeType::Interface ) {
                    throw CodeGenError("Intrinsic '" + func_name + "' not implemented yet and compile time value is not available.");
                } else {
                    h = get_hash((ASR::asr_t*)proc_sym);
                }
            }
        } else {
            throw CodeGenError("ABI type not implemented yet.");
        }
        if (llvm_symtab_fn_arg.find(h) != llvm_symtab_fn_arg.end()) {
            // Check if this is a callback function
            llvm::Value* fn = llvm_symtab_fn_arg[h];
            if (llvm_symtab_fn.find(h) == llvm_symtab_fn.end()) {
                throw CodeGenError("The callback function not found in llvm_symtab_fn");
            }
            llvm::FunctionType* fntype = llvm_symtab_fn[h]->getFunctionType();
            std::string m_name = std::string(((ASR::Function_t*)(&(x.m_name->base)))->m_name);
            args = convert_call_args(x, is_method);
            tmp = builder->CreateCall(fntype, fn, args);
        } else if (llvm_symtab_fn.find(h) == llvm_symtab_fn.end()) {
            throw CodeGenError("Function code not generated for '"
                + std::string(s->m_name) + "'");
        } else {
            llvm::Function *fn = llvm_symtab_fn[h];
            std::string m_name = std::string(((ASR::Function_t*)(&(x.m_name->base)))->m_name);
            std::vector<llvm::Value *> args2 = convert_call_args(x, is_method);
            args.insert(args.end(), args2.begin(), args2.end());
            if (pass_arg) {
                args.push_back(pass_arg);
            }
            ASR::ttype_t *return_var_type0 = EXPR2VAR(s->m_return_var)->m_type;
            if (ASRUtils::get_FunctionType(s)->m_abi == ASR::abiType::BindC) {
                if (is_a<ASR::Complex_t>(*return_var_type0)) {
                    int a_kind = down_cast<ASR::Complex_t>(return_var_type0)->m_kind;
                    if (a_kind == 8) {
                        if (compiler_options.platform == Platform::Windows) {
                            tmp = llvm_utils->CreateAlloca(*builder, complex_type_8);
                            args.insert(args.begin(), tmp);
                            builder->CreateCall(fn, args);
                            // Convert {double,double}* to {double,double}
                            tmp = llvm_utils->CreateLoad(tmp);
                        } else {
                            tmp = builder->CreateCall(fn, args);
                        }
                    } else {
                        tmp = builder->CreateCall(fn, args);
                    }
                } else {
                    tmp = builder->CreateCall(fn, args);
                }
            } else {
                tmp = CreateCallUtil(fn, args, return_var_type0);
            }
            // The convention we use is that any strings is a pointer to the underlying physicalType
            // Example of StringPhysicalTypes -> `string_descriptor*`, `i8*`
            if(ASRUtils::is_string_only(return_var_type0)){ 
                // Make sure to use `alloca` at the entry point.
                llvm::Value* string_ptr = llvm_utils->CreateAlloca(
                                            llvm_utils->get_StringType(return_var_type0),
                                            nullptr,
                                            "string_ret_const"
                                        );
                builder->CreateStore(tmp, string_ptr);
                tmp = string_ptr;
            }
        }
        if (ASRUtils::get_FunctionType(s)->m_abi == ASR::abiType::BindC) {
            ASR::ttype_t *return_var_type0 = EXPR2VAR(s->m_return_var)->m_type;
            if (is_a<ASR::Complex_t>(*return_var_type0)) {
                int a_kind = down_cast<ASR::Complex_t>(return_var_type0)->m_kind;
                if (a_kind == 4) {
                    if (compiler_options.platform == Platform::Windows) {
                        // tmp is i64, have to convert to {float, float}

                        // i64
                        llvm::Type* type_fx2 = llvm::Type::getInt64Ty(context);
                        // Convert i64 to i64*
                        llvm::AllocaInst *p_fx2 = llvm_utils->CreateAlloca(*builder, type_fx2);
                        builder->CreateStore(tmp, p_fx2);
                        // Convert i64* to {float,float}* using bitcast
                        tmp = builder->CreateBitCast(p_fx2, complex_type_4->getPointerTo());
                        // Convert {float,float}* to {float,float}
                        tmp = llvm_utils->CreateLoad(tmp);
                    } else if (compiler_options.platform == Platform::macOS_ARM) {
                        // pass
                    } else {
                        // tmp is <2 x float>, have to convert to {float, float}

                        // <2 x float>
                        llvm::Type* type_fx2 = FIXED_VECTOR_TYPE::get(llvm::Type::getFloatTy(context), 2);
                        // Convert <2 x float> to <2 x float>*
                        llvm::AllocaInst *p_fx2 = llvm_utils->CreateAlloca(*builder, type_fx2);
                        builder->CreateStore(tmp, p_fx2);
                        // Convert <2 x float>* to {float,float}* using bitcast
                        tmp = builder->CreateBitCast(p_fx2, complex_type_4->getPointerTo());
                        // Convert {float,float}* to {float,float}
                        tmp = llvm_utils->CreateLoad(tmp);
                    }
                }
            }
        }
        if (ASRUtils::is_character(*x.m_type)) {
            strings_to_be_deallocated.push_back(al, tmp);
        }
    }

    void load_array_size_deep_copy(ASR::expr_t* x) {
        if (x != nullptr &&  ASR::is_a<ASR::Var_t>(*x)) {
            ASR::Var_t* x_var = ASR::down_cast<ASR::Var_t>(x);
            ASR::symbol_t* x_sym = ASRUtils::symbol_get_past_external(x_var->m_v);
            if (x_sym != nullptr && ASR::is_a<ASR::Variable_t>(*x_sym)) {
                ASR::Variable_t* x_sym_variable = ASR::down_cast<ASR::Variable_t>(x_sym);
                uint32_t x_sym_variable_h = get_hash((ASR::asr_t*)x_sym_variable);
                if (llvm_symtab_deep_copy.find(x_sym_variable_h) != llvm_symtab_deep_copy.end()) {
                    tmp = llvm_utils->CreateLoad2(llvm_utils->get_type_from_ttype_t_util(x, ASRUtils::expr_type(x), module.get()),
                        llvm_symtab_deep_copy[x_sym_variable_h]);
                } else {
                    this->visit_expr_wrapper(x, true);
                }
            } else {
                this->visit_expr_wrapper(x, true);
            }
        } else {
            this->visit_expr_wrapper(x, true);
        }
    }

    void visit_ArraySizeUtil(ASR::expr_t* m_v, ASR::ttype_t* m_type,
        ASR::expr_t* m_dim=nullptr, ASR::expr_t* m_value=nullptr) {
        if( m_value ) {
            visit_expr_wrapper(m_value, true);
            return ;
        }

        int output_kind = ASRUtils::extract_kind_from_ttype_t(m_type);
        int dim_kind = 4;
        int64_t ptr_loads_copy = ptr_loads;
        ptr_loads = 2 - // Sync: instead of 2 - , should this be ptr_loads_copy -
                    LLVM::is_llvm_pointer(*ASRUtils::expr_type(m_v));
        visit_expr_wrapper(m_v);
        ptr_loads = ptr_loads_copy;
        ASR::ttype_t* x_mv_type = ASRUtils::expr_type(m_v);
        LCOMPILERS_ASSERT(ASRUtils::is_array(x_mv_type));
        llvm::Type* array_type = llvm_utils->get_type_from_ttype_t_util(m_v,
                ASRUtils::type_get_past_allocatable(ASRUtils::type_get_past_pointer(x_mv_type)), module.get());
        if (is_a<ASR::StructInstanceMember_t>(*m_v)) {
            tmp = llvm_utils->CreateLoad2(array_type->getPointerTo(), tmp);
        }
#if LLVM_VERSION_MAJOR > 16
            ptr_type[tmp] = array_type;
#endif
        llvm::Value* llvm_arg = tmp;

        llvm::Value* llvm_dim = nullptr;
        if( m_dim ) {
            visit_expr_wrapper(m_dim, true);
            dim_kind = ASRUtils::extract_kind_from_ttype_t(ASRUtils::expr_type(m_dim));
            llvm_dim = tmp;
        }

        ASR::array_physical_typeType physical_type = ASRUtils::extract_physical_type(x_mv_type);
        if (physical_type == ASR::array_physical_typeType::StringArraySinglePointer) {
            if (ASRUtils::is_fixed_size_array(x_mv_type)) {
                physical_type = ASR::array_physical_typeType::FixedSizeArray;
            } else {
                physical_type = ASR::array_physical_typeType::DescriptorArray;
            }
        }
        switch( physical_type ) {
            case ASR::array_physical_typeType::DescriptorArray: {
                tmp = arr_descr->get_array_size(llvm_arg, llvm_dim, output_kind, dim_kind);
                break;
            }
            case ASR::array_physical_typeType::PointerToDataArray:
            case ASR::array_physical_typeType::FixedSizeArray: {
                    llvm::Type* target_type = llvm_utils->get_type_from_ttype_t_util(m_v,
                        ASRUtils::type_get_past_allocatable(
                            ASRUtils::type_get_past_pointer(m_type)), module.get());


                    ASR::dimension_t* m_dims = nullptr;
                    int n_dims = ASRUtils::extract_dimensions_from_ttype(x_mv_type, m_dims);
                    if( llvm_dim ) {
                    llvm::AllocaInst *target = llvm_utils->CreateAlloca(
                        target_type, nullptr, "array_size");
                    llvm::BasicBlock *mergeBB = llvm::BasicBlock::Create(context, "ifcont");
                    for( int i = 0; i < n_dims; i++ ) {
                        llvm::Function *fn = builder->GetInsertBlock()->getParent();

                        llvm::BasicBlock *thenBB = llvm::BasicBlock::Create(context, "then", fn);
                        llvm::BasicBlock *elseBB = llvm::BasicBlock::Create(context, "else");

                        llvm::Value* cond = builder->CreateICmpEQ(llvm_dim,
                            llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), llvm::APInt(32, i + 1)));
                        builder->CreateCondBr(cond, thenBB, elseBB);
                        builder->SetInsertPoint(thenBB);
                        {
                            this->visit_expr_wrapper(m_dims[i].m_length, true);
                            builder->CreateStore(tmp, target);
                        }
                        builder->CreateBr(mergeBB);

                        start_new_block(elseBB);
                    }
                    start_new_block(mergeBB);
                    tmp = llvm_utils->CreateLoad(target);
                } else {
                    int kind = ASRUtils::extract_kind_from_ttype_t(m_type);
                    if( physical_type == ASR::array_physical_typeType::FixedSizeArray ) {
                        int64_t size = ASRUtils::get_fixed_size_of_array(m_dims, n_dims);
                        tmp = llvm::ConstantInt::get(target_type, llvm::APInt(8 * kind, size));
                    } else {
                        llvm::Value* llvm_size = llvm::ConstantInt::get(target_type, llvm::APInt(8 * kind, 1));
                        int ptr_loads_copy = ptr_loads;
                        ptr_loads = 2;
                        for( int i = 0; i < n_dims; i++ ) {
                            load_array_size_deep_copy(m_dims[i].m_length);

                            // Make dimension length and return size compatible.
                            if(ASRUtils::extract_kind_from_ttype_t(
                                ASRUtils::expr_type(m_dims[i].m_length)) > kind){
                                    tmp = builder->CreateTrunc(tmp, llvm::IntegerType::get(context, 8 * kind));
                            } else if (ASRUtils::extract_kind_from_ttype_t(
                                ASRUtils::expr_type(m_dims[i].m_length)) < kind){
                                tmp = builder->CreateSExt(tmp, llvm::IntegerType::get(context, 8 * kind));
                            }

                            llvm_size = builder->CreateMul(tmp, llvm_size);
                        }
                        ptr_loads = ptr_loads_copy;
                        tmp = llvm_size;
                    }
                }
                break;
            }
            default: {
                LCOMPILERS_ASSERT(false);
            }
        }
    }

    void visit_ArraySize(const ASR::ArraySize_t& x) {
        visit_ArraySizeUtil(x.m_v, x.m_type, x.m_dim, x.m_value);
    }

    void visit_ArrayBound(const ASR::ArrayBound_t& x) {
        ASR::expr_t* array_value = ASRUtils::expr_value(x.m_v);
        if( array_value && ASR::is_a<ASR::ArrayConstant_t>(*array_value) ) {
            ASR::ArrayConstant_t* array_const = ASR::down_cast<ASR::ArrayConstant_t>(array_value);
            int kind = ASRUtils::extract_kind_from_ttype_t(x.m_type);
            size_t bound_value = 0;
            if( x.m_bound == ASR::arrayboundType::LBound ) {
                bound_value = 1;
            } else if( x.m_bound == ASR::arrayboundType::UBound ) {
                bound_value = ASRUtils::get_fixed_size_of_array(array_const->m_type);
            } else {
                LCOMPILERS_ASSERT(false);
            }
            tmp = llvm::ConstantInt::get(context, llvm::APInt(kind * 8, bound_value));
            return ;
        }

        ASR::ttype_t* x_mv_type = ASRUtils::expr_type(x.m_v);
        llvm::Type* array_type = llvm_utils->get_type_from_ttype_t_util(x.m_v,
                ASRUtils::type_get_past_allocatable(ASRUtils::type_get_past_pointer(x_mv_type)), module.get());
        int64_t ptr_loads_copy = ptr_loads;
        ptr_loads = 2 - // Sync: instead of 2 - , should this be ptr_loads_copy -
                    (LLVM::is_llvm_pointer(*ASRUtils::expr_type(x.m_v)));
        visit_expr_wrapper(x.m_v);
        ptr_loads = ptr_loads_copy;
        if (is_a<ASR::StructInstanceMember_t>(*x.m_v)) {
            tmp = llvm_utils->CreateLoad2(array_type->getPointerTo(), tmp);
#if LLVM_VERSION_MAJOR > 16
            ptr_type[tmp] = array_type;
#endif
        }
        llvm::Value* llvm_arg1 = tmp;
        visit_expr_wrapper(x.m_dim, true);
        llvm::Value* dim_val = tmp;

        ASR::array_physical_typeType physical_type = ASRUtils::extract_physical_type(x_mv_type);
        switch( physical_type ) {
            case ASR::array_physical_typeType::DescriptorArray: {
                llvm::Value* dim_des_val = arr_descr->get_pointer_to_dimension_descriptor_array(array_type, llvm_arg1);
                llvm::Value* const_1 = llvm::ConstantInt::get(context, llvm::APInt(32, 1));
                dim_val = builder->CreateSub(dim_val, const_1);
                llvm::Value* dim_struct = arr_descr->get_pointer_to_dimension_descriptor(dim_des_val, dim_val);
                llvm::Value* res = nullptr;
                if( x.m_bound == ASR::arrayboundType::LBound ) {
                    res = arr_descr->get_lower_bound(dim_struct);
                } else if( x.m_bound == ASR::arrayboundType::UBound ) {
                    res = arr_descr->get_upper_bound(dim_struct);
                }
                tmp = res;
                break;
            }
            case ASR::array_physical_typeType::FixedSizeArray:
            case ASR::array_physical_typeType::PointerToDataArray: {
                llvm::Type* target_type = llvm_utils->get_type_from_ttype_t_util(x.m_v,
                    ASRUtils::type_get_past_allocatable(
                        ASRUtils::type_get_past_pointer(x.m_type)), module.get());
                llvm::AllocaInst *target = llvm_utils->CreateAlloca(
                    target_type, nullptr, "array_bound");
                llvm::BasicBlock *mergeBB = llvm::BasicBlock::Create(context, "ifcont");
                ASR::dimension_t* m_dims = nullptr;
                int n_dims = ASRUtils::extract_dimensions_from_ttype(x_mv_type, m_dims);
                for( int i = 0; i < n_dims; i++ ) {
                    llvm::Function *fn = builder->GetInsertBlock()->getParent();

                    llvm::BasicBlock *thenBB = llvm::BasicBlock::Create(context, "then", fn);
                    llvm::BasicBlock *elseBB = llvm::BasicBlock::Create(context, "else");

                    llvm::Value* cond = builder->CreateICmpEQ(dim_val,
                        llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), llvm::APInt(32, i + 1)));
                    builder->CreateCondBr(cond, thenBB, elseBB);
                    builder->SetInsertPoint(thenBB);
                    {
                        if( x.m_bound == ASR::arrayboundType::LBound ) {
                            this->visit_expr_wrapper(m_dims[i].m_start, true);
                            tmp = builder->CreateSExtOrTrunc(tmp, target_type);
                            builder->CreateStore(tmp, target);
                        } else if( x.m_bound == ASR::arrayboundType::UBound ) {
                            llvm::Value *lbound = nullptr, *length = nullptr;
                            this->visit_expr_wrapper(m_dims[i].m_start, true);
                            lbound = tmp;
                            load_array_size_deep_copy(m_dims[i].m_length);
                            length = tmp;
                            builder->CreateStore(
                                builder->CreateSub(builder->CreateSExtOrTrunc(builder->CreateAdd(length, lbound), target_type),
                                      llvm::ConstantInt::get(context, llvm::APInt(32, 1))),
                                target);
                        }
                    }
                    builder->CreateBr(mergeBB);

                    start_new_block(elseBB);
                }
                start_new_block(mergeBB);
                tmp = llvm_utils->CreateLoad2(target_type, target);
                break;
            }
            case ASR::array_physical_typeType::SIMDArray: {
                if( x.m_bound == ASR::arrayboundType::LBound ) {
                    tmp = llvm::ConstantInt::get(context, llvm::APInt(32, 1));
                } else if( x.m_bound == ASR::arrayboundType::UBound ) {
                    int64_t size = ASRUtils::get_fixed_size_of_array(ASRUtils::expr_type(x.m_v));
                    tmp = llvm::ConstantInt::get(context, llvm::APInt(32, size));
                }
                break;
            }
            case ASR::array_physical_typeType::StringArraySinglePointer: {
                ASR::dimension_t* m_dims = nullptr;
                int n_dims = ASRUtils::extract_dimensions_from_ttype(x_mv_type, m_dims);
                if (ASRUtils::is_dimension_empty(m_dims, n_dims)) {
                    // treat it as DescriptorArray
                    llvm::Value* dim_des_val = arr_descr->get_pointer_to_dimension_descriptor_array(llvm_arg1);
                    llvm::Value* const_1 = llvm::ConstantInt::get(context, llvm::APInt(32, 1));
                    dim_val = builder->CreateSub(dim_val, const_1);
                    llvm::Value* dim_struct = arr_descr->get_pointer_to_dimension_descriptor(dim_des_val, dim_val);
                    llvm::Value* res = nullptr;
                    if( x.m_bound == ASR::arrayboundType::LBound ) {
                        res = arr_descr->get_lower_bound(dim_struct);
                    } else if( x.m_bound == ASR::arrayboundType::UBound ) {
                        res = arr_descr->get_upper_bound(dim_struct);
                    }
                    tmp = res;
                    break;
                } else if (ASRUtils::is_fixed_size_array(x_mv_type)) {
                    llvm::Type* target_type = llvm_utils->get_type_from_ttype_t_util(x.m_v,
                        ASRUtils::type_get_past_allocatable(
                            ASRUtils::type_get_past_pointer(x.m_type)), module.get());
                    llvm::AllocaInst *target = llvm_utils->CreateAlloca(
                        target_type, nullptr, "array_bound");
                    llvm::BasicBlock *mergeBB = llvm::BasicBlock::Create(context, "ifcont");
                    ASR::dimension_t* m_dims = nullptr;
                    int n_dims = ASRUtils::extract_dimensions_from_ttype(x_mv_type, m_dims);
                    for( int i = 0; i < n_dims; i++ ) {
                        llvm::Function *fn = builder->GetInsertBlock()->getParent();

                        llvm::BasicBlock *thenBB = llvm::BasicBlock::Create(context, "then", fn);
                        llvm::BasicBlock *elseBB = llvm::BasicBlock::Create(context, "else");

                        llvm::Value* cond = builder->CreateICmpEQ(dim_val,
                            llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), llvm::APInt(32, i + 1)));
                        builder->CreateCondBr(cond, thenBB, elseBB);
                        builder->SetInsertPoint(thenBB);
                        {
                            if( x.m_bound == ASR::arrayboundType::LBound ) {
                                this->visit_expr_wrapper(m_dims[i].m_start, true);
                                builder->CreateStore(tmp, target);
                            } else if( x.m_bound == ASR::arrayboundType::UBound ) {
                                llvm::Value *lbound = nullptr, *length = nullptr;
                                this->visit_expr_wrapper(m_dims[i].m_start, true);
                                lbound = tmp;
                                load_array_size_deep_copy(m_dims[i].m_length);
                                length = tmp;
                                builder->CreateStore(
                                    builder->CreateSub(builder->CreateAdd(length, lbound),
                                        llvm::ConstantInt::get(context, llvm::APInt(32, 1))),
                                    target);
                            }
                        }
                        builder->CreateBr(mergeBB);

                        start_new_block(elseBB);
                    }
                    start_new_block(mergeBB);
                    tmp = llvm_utils->CreateLoad2(target_type, target);
                    break;
                } else {
                    LCOMPILERS_ASSERT(false);
                    break;
                }
            }
            default: {
                LCOMPILERS_ASSERT(false);
            }
        }
    }

    void visit_StringFormat(const ASR::StringFormat_t& x) {
        // TODO: Handle some things at compile time if possible:
        //ASR::expr_t* fmt_value = ASRUtils::expr_value(x.m_fmt);
        // if (fmt_value) ...
        if (x.m_kind == ASR::string_format_kindType::FormatFortran) {
            std::vector<llvm::Value *> args;
            // Push fmt string.
            if(x.m_fmt == nullptr){ // default formatting
                llvm::Type* int8Type = builder->getInt8Ty();
                llvm::PointerType* charPtrType = int8Type->getPointerTo();
                llvm::Constant* nullCharPtr = llvm::ConstantPointerNull::get(charPtrType);
                args.push_back(nullCharPtr);
                args.push_back(llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), 0));
            } else {
                llvm::Value* fmt_data, *fmt_len;
                std::tie(fmt_data, fmt_len) = get_string_data_and_length(x.m_fmt);
                args.push_back(fmt_data);
                args.push_back(fmt_len);
            }
            // Push Serialization;
            llvm::Value* serialization_info = SerializeExprTypes(x.m_args, x.n_args);
            args.push_back(serialization_info);

            //Push serialization of sizes and n_size
            size_t ArraySizesCnt = 0;
            for (size_t i=0; i<x.n_args; i++) {
                if(ASRUtils::is_array(ASRUtils::expr_type(x.m_args[i]))){
                    ASR::expr_t* ArraySizeExpr = ASRUtils::EXPR
                        (ASR::make_ArraySize_t(al, x.m_args[i]->base.loc, x.m_args[i],
                        nullptr, ASRUtils::TYPE(ASR::make_Integer_t(al, x.m_args[i]->base.loc, 8)),
                        ASRUtils::get_compile_time_array_size(al,
                            ASRUtils::expr_type(x.m_args[i])))) ;
                    visit_expr(*ArraySizeExpr);
                    args.push_back(tmp);
                    ArraySizesCnt++;
                }
            }

            args.insert(args.end()-ArraySizesCnt,
                llvm::ConstantInt::get(llvm::Type::getInt32Ty(context),
                    ArraySizesCnt));
            // Push runtime-length of strings and n_length
            size_t string_lengths_cnt = 0;
            for(size_t i=0; i<x.n_args; i++) {
                // Only loop on string types.
                if(!ASRUtils::is_character(*
                    ASRUtils::extract_type(
                        expr_type(x.m_args[i])))){continue;}

                ASR::String_t* str_type =
                    ASR::down_cast<ASR::String_t>(
                        ASRUtils::extract_type(
                            expr_type(x.m_args[i])));
                int64_t len; // dummy to use below.
                if(ASRUtils::extract_value(str_type->m_len, len)){
                    // It's serialized at compile time (`S-len`).
                } else {
                    args.push_back(get_string_length(x.m_args[i]));
                    ++string_lengths_cnt;
                }
            }
            args.insert(args.end()- string_lengths_cnt - ArraySizesCnt,
                llvm::ConstantInt::get(llvm::Type::getInt32Ty(context),
                    string_lengths_cnt));

            // Push the args.
            for (size_t i=0; i<x.n_args; i++) {
                // Push the args as raw pointers
                int64_t ptr_load_copy = ptr_loads;
                ptr_loads = (ASR::is_a<ASR::Var_t>(*x.m_args[i]) && !ASRUtils::is_character(*expr_type(x.m_args[i]))) ? 0 : 1;
                // Special Hanlding to pass appropriate pointer to the backend.
                if(ASRUtils::is_array(expr_type(x.m_args[i]))){ // Arrays need a cast to pointerToDataArray physicalType.
                    ASR::Array_t* arr = ASR::down_cast<ASR::Array_t>(
                        ASRUtils::type_get_past_allocatable_pointer(
                            ASRUtils::expr_type(x.m_args[i])));
                    if(arr->m_physical_type != ASR::array_physical_typeType::PointerToDataArray){
                        ASR::ttype_t* array_type = ASRUtils::TYPE(
                                                    ASR::make_Array_t(al, arr->base.base.loc,arr->m_type,
                                                    arr->m_dims, arr->n_dims,ASR::array_physical_typeType::FixedSizeArray));
                        ASR::expr_t* array_casted_to_pointer = ASRUtils::EXPR(
                                                                ASR::make_ArrayPhysicalCast_t(al, arr->base.base.loc,
                                                                x.m_args[i],arr->m_physical_type,
                                                                ASR::array_physical_typeType::PointerToDataArray,
                                                                array_type, nullptr));
                        this->visit_expr(*array_casted_to_pointer);
                    } else {
                        this->visit_expr(*x.m_args[i]);
                    }
                } else if (ASRUtils::is_character(*expr_type(x.m_args[i])) && 
                    ASRUtils::get_string_type(expr_type(x.m_args[i]))->m_physical_type == ASR::CChar) {
                    visit_expr_load_wrapper(x.m_args[i], 0);
                    LCOMPILERS_ASSERT(llvm_utils->get_StringType(expr_type(x.m_args[i])) == llvm::Type::getInt8Ty(context))
                    LCOMPILERS_ASSERT(tmp->getType()->isPointerTy()); // atleast not `char`
                    llvm::Value* tmp_ptr = builder->CreateAlloca(llvm::Type::getInt8Ty(context)->getPointerTo()->getPointerTo());
                    builder->CreateStore(tmp, tmp_ptr);
                    tmp = tmp_ptr;
                } else {
                    ptr_loads = ptr_loads + LLVM::is_llvm_pointer(*expr_type(x.m_args[i]));
                    this->visit_expr_wrapper(x.m_args[i], true);
                }
                if(!tmp->getType()->isPointerTy() ||
                    ASR::is_a<ASR::PointerToCPtr_t>(*x.m_args[i])){
                    llvm::Value* tmp_ptr = builder->CreateAlloca(tmp->getType());
                    builder->CreateStore(tmp, tmp_ptr);
                    tmp = tmp_ptr;
                }
                args.push_back(tmp);
                ptr_loads = ptr_load_copy;
            }
            tmp = string_format_fortran(context, *module, *builder, args);
            tmp = llvm_utils->create_string_descriptor(tmp, lfortran_str_len(tmp),"stringFormat_desc");
        } else {
            throw CodeGenError("Only FormatFortran string formatting implemented so far.");
        }
    }

    void visit_ArrayBroadcast(const ASR::ArrayBroadcast_t &x) {
        this->visit_expr_wrapper(x.m_array, true);
        llvm::Value *value = tmp;
        llvm::Type* ele_type = llvm_utils->get_type_from_ttype_t_util(x.m_array,
            ASRUtils::type_get_past_array(x.m_type), module.get());
        size_t n_eles = ASRUtils::get_fixed_size_of_array(x.m_type);
        llvm::Type* vec_type = FIXED_VECTOR_TYPE::get(ele_type, n_eles);
        llvm::AllocaInst *vec = llvm_utils->CreateAlloca(*builder, vec_type);
        for (size_t i=0; i < n_eles; i++) {
            builder->CreateStore(value, llvm_utils->create_gep2(vec_type, vec, i));
        }
        tmp = llvm_utils->CreateLoad2(vec_type, vec);
    }
};



Result<std::unique_ptr<LLVMModule>> asr_to_llvm(ASR::TranslationUnit_t &asr,
        diag::Diagnostics &diagnostics,
        llvm::LLVMContext &context, Allocator &al,
        LCompilers::PassManager& pass_manager,
        CompilerOptions &co, const std::string &run_fn, const std::string &/*global_underscore*/,
        const std::string &infile)
{
#if LLVM_VERSION_MAJOR >= 15 && LLVM_VERSION_MAJOR <= 19
    context.setOpaquePointers(false);
#endif
    ASRToLLVMVisitor v(al, context, infile, co, diagnostics);

    std::vector<int64_t> skip_optimization_func_instantiation;
    skip_optimization_func_instantiation.push_back(static_cast<int64_t>(
                    ASRUtils::IntrinsicElementalFunctions::FlipSign));
    skip_optimization_func_instantiation.push_back(static_cast<int64_t>(
                    ASRUtils::IntrinsicElementalFunctions::FMA));
    skip_optimization_func_instantiation.push_back(static_cast<int64_t>(
                    ASRUtils::IntrinsicElementalFunctions::SignFromValue));

    co.po.run_fun = run_fn;
    co.po.always_run = false;
    co.po.skip_optimization_func_instantiation = skip_optimization_func_instantiation;
    pass_manager.rtlib = co.rtlib;
    auto t1 = std::chrono::high_resolution_clock::now();
    pass_manager.apply_passes(al, &asr, co.po, diagnostics);
    auto t2 = std::chrono::high_resolution_clock::now();

    if (co.time_report) {
        int time_take_to_run_asr_passes = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        double time_in_milliseconds = (double) time_take_to_run_asr_passes / 1000.0;
        std::string message = "";
        if ( time_in_milliseconds >= 1.0 ) {
            message = "ASR -> ASR passes: " + std::to_string(time_take_to_run_asr_passes / 1000) + "." + std::to_string(time_take_to_run_asr_passes % 1000) + " ms";
        } else {
            message = "ASR -> ASR passes: " + std::to_string(time_in_milliseconds) + " ms";
        }
        co.po.vector_of_time_report.push_back(message);
    }

    // Uncomment for debugging the ASR after the transformation
    // std::cout << LCompilers::pickle(asr, true, false, false) << std::endl;

    t1 = std::chrono::high_resolution_clock::now();
    try {
        v.visit_asr((ASR::asr_t&)asr);
    } catch (const CodeGenError &e) {
        Error error;
        diagnostics.diagnostics.push_back(e.d);
        return error;
    } catch (const CodeGenAbort &) {
        LCOMPILERS_ASSERT(diagnostics.has_error())
        Error error;
        return error;
    }
    std::string msg;
    llvm::raw_string_ostream err(msg);
    if (llvm::verifyModule(*v.module, &err)) {
        std::string buf;
        llvm::raw_string_ostream os(buf);
        v.module->print(os, nullptr);
        std::cout << os.str();
        msg = "asr_to_llvm: module failed verification. Error:\n" + err.str();
        std::cout << msg << std::endl;
        diagnostics.diagnostics.push_back(diag::Diagnostic(msg,
            diag::Level::Error, diag::Stage::CodeGen));
        Error error;
        return error;
    };

    std::unique_ptr<LLVMModule> res = std::make_unique<LLVMModule>(std::move(v.module));
    t2 = std::chrono::high_resolution_clock::now();

    if (co.time_report) {
        int time_take_to_generate_llvm_ir = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        std::string message = "LLVM IR creation: " + std::to_string(time_take_to_generate_llvm_ir / 1000) + "." + std::to_string(time_take_to_generate_llvm_ir % 1000) + " ms";
        co.po.vector_of_time_report.push_back(message);
    }

    return res;
}

} // namespace LCompilers
