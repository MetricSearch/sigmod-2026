classdef min_heap
    %MIN_HEAP undefined
    %   undefined

    properties
        list
        list_end_ptr
        max_value
        min_value
    end

    methods
        function obj = min_heap()
            %MIN_HEAP undefined
            %   undefined
            obj.list = zeros(1,10);
            obj.list_end_ptr = 1;
            obj.min_value = -1;
            obj.max_value = -1;
        end

        function b = is_empty(obj)
            %METHOD1 undefined
            %   undefined
            b = obj.list_end_ptr == 1;
        end

        function obj = push(obj,v)
            %METHOD1 undefined
            %   undefined
            if sum(obj.list == v) == 0
                %only add this value if it's not already in the list
                obj.list(obj.list_end_ptr) = v;
                obj.list_end_ptr = obj.list_end_ptr + 1;
                if obj.list_end_ptr > length(obj.list)
                    obj.list = [obj.list,[0,0,0,0,0]];
                end
                if obj.min_value == -1
                    obj.min_value = v;
                else
                    obj.min_value = min(obj.min_value,v);
                end
                obj.max_value = max(obj.max_value,v);
            end
        end

        function [obj,v] = pop(obj)
            %METHOD1 undefined
            %   undefined
            %return minimum
            v = obj.min_value;
            %find its first instance in the list and shift the rhs down one
            wh = find(obj.list == v);
            if isempty(wh)
                disp(obj);
            end
            % obj.list(wh(1):obj.list_end_ptr - 1) = obj.list(wh(1)+1:obj.list_end_ptr);
            obj.list(wh(1):end-1) = obj.list(wh(1)+1:end);
            obj.list_end_ptr = obj.list_end_ptr - 1;

            %reset minimum
            if obj.list_end_ptr > 2
                obj.min_value = min(obj.list(1:obj.list_end_ptr-1));
            else
                if obj.list_end_ptr > 1
                    obj.min_value = obj.list(1);
                else
                    obj.min_value = -1;
                    obj.max_value = -1;
                end
            end
        end

    end %of methods

end % of object